import json
import traceback
from typing import List, Dict
from search import CourseRAGManager
from config_manager import load_config
from core import generate_document_id
from langfuse.decorators import observe, langfuse_context
from typing import List, Dict, Any

def generate_markdown_report(results: List[Dict[str, Any]], filename: str = "eval_diagnostic.md"):
    with open(filename, "w") as f:
        f.write("# 🔍 RAG Retrieval Diagnostic Report\n\n")
        
        for i, r in enumerate(results):
            status = "✅ PASS" if r["success"] else "❌ FAIL"
            f.write(f"### {i+1}. {status} | Course: {r['course']}\n")
            f.write(f"**Query:** `{r['query']}`\n\n")
            
            if not r["success"]:
                f.write("#### 🎯 Expected (What you wanted)\n")
                f.write(f"- **ID:** `{r['expected_id'][:8]}`\n")
                f.write(f"> {r['expected_text'][:250]}...\n\n")
                
                f.write(f"#### 🚫 Found at Rank #1 (Score: {r['found_score']:.2f})\n")
                f.write(f"- **ID:** `{r['found_id'][:8]}`\n")
                f.write(f"> {r['best_text'][:250]}...\n\n")
            else:
                f.write(f"**Match Found** (Score: {r['found_score']:.2f})\n\n")
            f.write("---\n\n")









def get_deterministic_eval_set(filepath: str, n_per_course: int = 10) -> List[Dict]:
    with open(filepath, 'r') as f:
        raw_data = json.load(f)
    
    eval_set = []
    for course_entry in raw_data:
        course_name = course_entry["course"]
        for i, doc in enumerate(course_entry["documents"]):
            if i >= n_per_course:
                break
            
            clean_doc = {
                "text": doc["text"],
                "question": doc["question"],
                "course": course_name
            }
            
            eval_set.append({
                "query": doc["question"],
                "course": course_name,
                "expected_id": generate_document_id(clean_doc),
                "expected_text": doc["text"]  # Store this for the report!
            })
    return eval_set


@observe()
def calculate_recall(manager: CourseRAGManager, eval_set: List[Dict], k: int) -> Dict[str, Any]:
    hits = 0
    details = []
    
    for item in eval_set:
        # Note: Using your current global search (no course filter)
        results = manager.search_faq(item["query"], override_size=k)
        
        # Metadata for Rank #1
        top_hit = results[0] if results else None
        found_id = top_hit["_id"] if top_hit else "NONE"
        found_score = top_hit["_score"] if top_hit else 0.0
        best_text = top_hit["_source"]["text"] if top_hit else "NOTHING FOUND"
        
        retrieved_ids = [hit["_id"] for hit in results]
        success = item["expected_id"] in retrieved_ids
        
        if success:
            hits += 1
            
        details.append({
            "course": item["course"],
            "query": item["query"],
            "expected_id": item["expected_id"],
            "expected_text": item["expected_text"],
            "found_id": found_id,
            "found_score": found_score,
            "best_text": best_text,
            "success": success
        })
            
    recall_score = hits / len(eval_set)
    langfuse_context.update_current_trace(metadata={"k": k, "recall": recall_score})
    return {"recall": recall_score, "details": details}



@observe()
def run_evaluation_cycle():
    es_settings = load_config("settings_es.json")
    

    eval_set = get_deterministic_eval_set("documents.json", n_per_course=10)
    
    manager = CourseRAGManager(es_settings)
    manager.connect_elasticsearch()
    
    final_report_data = []
    
    k_values = [1, 5]
    for k in k_values:
        result = calculate_recall(manager, eval_set, k)
        print(f"Top-K: {k} | Recall: {result['recall']:.2%}")
        if k == 5:
            final_report_data = result["details"]
            
    generate_markdown_report(final_report_data)

if __name__ == "__main__":
    run_evaluation_cycle()
