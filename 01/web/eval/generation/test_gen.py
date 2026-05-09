import sys
import os
import re
import json
import asyncio
import random
import logging
from datetime import datetime
from difflib import SequenceMatcher
from collections import defaultdict
from dotenv import load_dotenv
from litellm import acompletion
from qdrant_client import QdrantClient

# --- CONFIGURATION & LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.insert(0, project_root)

load_dotenv('configs/.env')
os.environ["NVIDIA_NIM_API_KEY"] = os.getenv("NVIDIA_API_KEY")

from eval.judges.shared import JUDGE_MODEL_70B
MODEL = JUDGE_MODEL_70B

# --- HELPER FUNCTIONS ---

def load_prompts():
    logger.info("Loading prompts.json...")
    with open('eval/generation/prompts.json') as f:
        return json.load(f)

def get_diverse_sample(client, n=5):
    logger.info("Fetching documents from Qdrant...")
    all_docs = []
    offset = None 
    
    try:
        max_scrolls = 10 
        scroll_count = 0
        while scroll_count < max_scrolls:
            results = client.scroll(
                collection_name='faqs', 
                limit=100, 
                offset=offset, 
                with_payload=True
            )
            points, next_offset = results
            if not points:
                break
            for point in points:
                all_docs.append(point.payload)
            if next_offset is None:
                break
            offset = next_offset
            scroll_count += 1
    except Exception as e:
        logger.error(f"Failed to scroll Qdrant: {e}")
        return []

    groups = defaultdict(list)
    for doc in all_docs:
        groups[(doc.get('course', 'Unknown'), doc.get('section', ''))].append(doc)

    random.seed(42)
    sampled = []
    group_list = list(groups.values())
    random.shuffle(group_list)
    for group in group_list:
        if len(sampled) >= n: break
        if group: sampled.append(random.choice(group))
    
    return sampled[:n]

def similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def build_qa_pairs(docs):
    parts = []
    for i, doc in enumerate(docs, 1):
        parts.append(
            f"FAQ {i}:\n"
            f"Course: {doc['course']}\n"
            f"Question: {doc['question']}\n"
            f"Answer: {doc['answer'][:400]}\n"
        )
    return "\n".join(parts)

# --- MAIN EXECUTION ---

async def main():
    try:
        prompts = load_prompts()
        client = QdrantClient('localhost', port=6333)
        docs = get_diverse_sample(client, 5)
        
        if not docs:
            logger.error("No documents found.")
            return

        print("\n" + "=" * 70)
        print("DOCUMENTS SAMPLED:")
        for i, doc in enumerate(docs, 1):
            print(f"  {i}. [{doc['course']}] {doc['question'][:80]}")
        print("=" * 70)
        
        prompt_name = "natural_diversity"
        template = prompts[prompt_name]['template']
        qa_pairs = build_qa_pairs(docs)
        prompt = template.replace("{qa_pairs}", qa_pairs)
        
        logger.info(f"Sending request to {MODEL}...")
        start_time = datetime.now()
        
        response = await acompletion(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1000,
        )
        
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Response received in {duration:.2f}s.")
        
        raw = response.choices[0].message.content.strip()
        
        print("\n" + "################" * 5)
        print("RAW MODEL OUTPUT:")
        print(raw)
        print("################" * 5 + "\n")
        
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not json_match:
            logger.error("Could not find JSON object in response.")
            return
            
        result = json.loads(json_match.group())
        
        print("--- PARSED & GRANULAR DIVERSITY CHECK ---")
        total_queries = 0
        total_kept = 0

        for i, doc in enumerate(docs, 1):
            variations = result.get(str(i), [])
            print(f"\n  FAQ {i}: {doc['question'][:70]}")
            
            if not variations:
                print("    NO QUERIES RETURNED")
                continue
            
            faq_kept = 0
            for j, q in enumerate(variations):
                total_queries += 1
                sim = similarity(q, doc['question'])
                is_diverse = sim < 0.6
                status = "✓" if is_diverse else "✗ TOO SIMILAR"
                
                if is_diverse:
                    total_kept += 1
                    faq_kept += 1
                
                print(f"    [{j+1}] Sim: {sim:.0%} | {status} | {q}")
            
            print(f"    → Local Score: {faq_kept}/{len(variations)} diverse")
        
        print(f"\n{'='*70}")
        print(f"FINAL STATS")
        print(f"Total Queries: {total_queries}")
        print(f"Passed Filter: {total_kept}")
        print(f"Pass Rate:     {(total_kept/total_queries if total_queries > 0 else 0):.1%}")
        print(f"{'='*70}")
            
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    asyncio.run(main())
