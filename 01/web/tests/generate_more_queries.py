import sys
sys.path.insert(0, '/home/admin/LLM/LLM/01/web')
import json
import litellm
from tqdm import tqdm

def generate_paraphrases(original_question, num_variations=5):
    prompt = f"""Generate {num_variations} alternative ways a user might ask this question. Use natural, conversational language. Change wording and sentence structure. Do NOT use the original wording. Keep each under 15 words. Output one per line, numbered.

Original: {original_question}

Variations:"""
    
    try:
        response = litellm.completion(
            model="nvidia_nim/nvidia/nemotron-mini-4b-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200
        )
        result = response.choices[0].message.content
        lines = result.strip().split('\n')
        paraphrases = []
        for line in lines:
            cleaned = line.strip()
            if cleaned and cleaned[0].isdigit():
                cleaned = cleaned.split('.', 1)[-1].strip()
                cleaned = cleaned.split(')', 1)[-1].strip()
            if cleaned and len(cleaned) > 5:
                paraphrases.append(cleaned)
        return paraphrases[:num_variations]
    except Exception as e:
        print(f"Generation error: {e}")
        return []

def main():
    print("=" * 60)
    print("GENERATING ADDITIONAL TEST QUERIES")
    print("=" * 60)
    
    # Load original questions
    with open('../documents.json', 'r') as f:
        data = json.load(f)
    
    # Collect all original questions
    original_questions = []
    for course in data:
        for doc in course['documents']:
            question = doc['question']
            if " - " in question:
                question = question.split(" - ", 1)[1].strip()
            original_questions.append({
                'question': question,
                'course': course['course'],
                'text': doc['text']
            })
    
    print(f"📊 Found {len(original_questions)} original questions")
    
    # Generate 5 new paraphrases for each of the first 20 questions
    new_queries = []
    for item in tqdm(original_questions[:20], desc="Generating"):
        paraphrases = generate_paraphrases(item['question'], num_variations=5)
        for p in paraphrases:
            new_queries.append({
                'query': p,
                'expected_course': item['course'],
                'original_question': item['question'],
                'expected_text': item['text']
            })
    
    # Save expanded eval set
    output_path = '../experiments/expanded_eval_set.json'
    with open(output_path, 'w') as f:
        json.dump(new_queries, f, indent=2)
    
    print(f"\n✅ Generated {len(new_queries)} new test queries")
    print(f"   Saved to {output_path}")
    print(f"\n📊 New total queries: {len(new_queries)}")

if __name__ == "__main__":
    main()
