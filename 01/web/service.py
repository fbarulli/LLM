"""
Two-stage FAQ Service: CAG (fast cache) → RAG (LLM fallback)
- Async, non-blocking
- Glossary query expansion with hit tracking
- Course boosting (overfetch + re-rank)
- Atomic CAG persistence
- Guardrails: low confidence → "I don't know"

Run:    uv run python service.py
Test:   curl -X POST http://localhost:7870/ask -H 'Content-Type: application/json' -d '{"question":"how do I turn words into numbers?"}'
"""
import sys, os, json, tempfile, shutil
from pathlib import Path
from collections import Counter

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))

from dotenv import load_dotenv
load_dotenv(BASE / 'configs/.env')
os.environ["NVIDIA_NIM_API_KEY"] = os.getenv("NVIDIA_API_KEY")

from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from litellm import acompletion

# ── Config ──────────────────────────────────────────────────────────────────
MODEL = "nvidia_nim/meta/llama-3.1-70b-instruct"
EMBEDDING = 'BAAI/bge-base-en-v1.5'
COLLECTION = 'faqs_bge_base_en_v1.5'
CAG_PATH = BASE / 'experiments/cag_answers_v2.json'
CAG_THRESHOLD = 0.75
LOW_CONFIDENCE_THRESHOLD = 0.62
TOP_K = 5
OVERFETCH = 50

GLOSSARY = {
    'embedding': ['turn words into numbers', 'word vectors', 'text to numbers'],
    'encoding error': ['csv weird bytes', 'weird characters'],
    'wget': ['downloader thing', 'download tool', 'install downloader'],
    'correlation matrix': ['correlating numerical and categorical'],
    'multicollinearity': ['prevent multicollinearity'],
    'docker build cache': ['docker model not updating', 'docker same result'],
}

# ── Load once ───────────────────────────────────────────────────────────────
print("Loading...")
embed_model = SentenceTransformer(EMBEDDING)
client = QdrantClient('localhost', port=6333)
expansion_hits = Counter()

def load_cag():
    if CAG_PATH.exists():
        with open(CAG_PATH) as f:
            return json.load(f)['answers']
    return {}

def save_cag(cag):
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, dir=CAG_PATH.parent)
    json.dump({"answers": cag}, tmp, indent=2)
    tmp.close()
    shutil.move(tmp.name, CAG_PATH)

cag = load_cag()
print(f"Loaded {len(cag)} CAG answers")

RAG_PROMPT = """You are a course teaching assistant. Answer the question using ONLY the FAQ contexts below.
If the contexts don't contain enough information, say exactly: "I don't have enough information to answer this question."

Question: {question}

Contexts:
{contexts}

Answer:"""

# ── Helpers ─────────────────────────────────────────────────────────────────
def expand_query(query: str) -> str:
    query_lower = query.lower()
    expansions = []
    for technical_term, colloquial_terms in GLOSSARY.items():
        if any(phrase in query_lower for phrase in colloquial_terms):
            expansions.append(technical_term)
            expansion_hits[technical_term] += 1
    if expansions:
        return query + ' ' + ' '.join(expansions)
    return query

def compute_boosted_score(hit, course: str = None) -> float:
    score = hit.score
    if course and hit.payload.get('course') == course:
        score *= 1.2
    return score

# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="FAQ Service")

class Query(BaseModel):
    question: str
    course: str = None

class Response(BaseModel):
    answer: str
    stage: str
    reason: str = None
    retrieval_score: float = None
    sources: list = []

@app.post("/ask", response_model=Response)
async def ask(q: Query):
    expanded = expand_query(q.question)
    vec = embed_model.encode(expanded).tolist()
    
    results = client.query_points(
        collection_name=COLLECTION, query=vec, limit=OVERFETCH, with_payload=True,
    )
    
    if not results.points:
        return Response(
            answer="I couldn't find any relevant answers. Try rephrasing.",
            stage="none", reason="no_results", retrieval_score=0.0,
        )
    
    # Sort by boosted score
    results.points.sort(key=lambda h: compute_boosted_score(h, q.course), reverse=True)
    results.points = results.points[:TOP_K]
    
    top_hit = results.points[0]
    top_score = compute_boosted_score(top_hit, q.course)
    top_id = top_hit.payload.get('es_id', '')
    
    # Guardrail
    if top_score < LOW_CONFIDENCE_THRESHOLD:
        return Response(
            answer="I don't have enough information to answer this question.",
            stage="none", reason="low_confidence", retrieval_score=top_score,
        )
    
    # Stage 1: CAG
    if top_id in cag and top_score >= CAG_THRESHOLD:
        cached = cag[top_id]
        return Response(
            answer=cached['generated_answer'], stage="cag", reason="cached",
            retrieval_score=top_score,
            sources=[{'question': top_hit.payload.get('question',''), 'course': top_hit.payload.get('course',''), 'score': top_score}],
        )
    
    # Stage 2: RAG
    contexts, sources = [], []
    for hit in results.points:
        p = hit.payload
        contexts.append(f"Q: {p.get('question','')}\nA: {p.get('answer','')[:500]}")
        sources.append({'question': p.get('question',''), 'course': p.get('course',''), 'score': compute_boosted_score(hit, q.course)})
    
    prompt = RAG_PROMPT.format(question=q.question, contexts="\n\n".join(contexts))
    
    try:
        resp = await acompletion(model=MODEL, messages=[{"role":"user","content":prompt}], temperature=0.3, max_tokens=500)
        answer = resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"RAG error: {e}")
        answer = "Error generating answer. Please try again."
    
    # Cache good answers
    if top_id not in cag and "don't have enough" not in answer.lower():
        cag[top_id] = {
            'question': top_hit.payload.get('question',''),
            'original_answer': top_hit.payload.get('answer',''),
            'generated_answer': answer,
            'course': top_hit.payload.get('course',''),
        }
        save_cag(cag)
    
    return Response(
        answer=answer, stage="rag",
        reason="generated" if "don't have enough" not in answer.lower() else "insufficient_context",
        retrieval_score=top_score, sources=sources,
    )

@app.get("/stats")
def stats():
    return {"cag_size": len(cag), "expansion_hits": dict(expansion_hits)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7870)