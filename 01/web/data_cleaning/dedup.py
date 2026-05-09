"""
data_cleaning/dedup.py
======================
Removes duplicate documents based on (question, answer_prefix).

Input:  data_cleaning/data/processed/parsed.jsonl
Output: data_cleaning/data/processed/clean.jsonl
        Deduplicated, one JSON object per line

Run:    uv run python data_cleaning/dedup.py
"""
import os
import json
from typing import Dict

INPUT = 'data_cleaning/data/processed/parsed.jsonl'
OUTPUT = 'data_cleaning/data/processed/clean.jsonl'


def main():
    seen = {}  # (question_lower, answer_prefix) -> doc
    duplicates = []
    total = 0

    with open(INPUT) as f:
        for line in f:
            doc = json.loads(line)
            total += 1
            
            key = (
                doc['question'].strip().lower(),
                doc['answer'][:200].strip().lower(),
            )
            
            if key in seen:
                duplicates.append({
                    'kept': seen[key]['id'],
                    'removed': doc['id'],
                    'question': doc['question'][:80],
                })
            else:
                seen[key] = doc

    with open(OUTPUT, 'w') as f:
        for doc in seen.values():
            f.write(json.dumps(doc) + '\n')

    print(f'Input:  {total} documents')
    print(f'Output: {len(seen)} documents')
    print(f'Duplicates removed: {len(duplicates)}')
    
    if duplicates:
        print('\nRemoved duplicates:')
        for d in duplicates:
            print(f"  {d['removed']} (kept {d['kept']}): {d['question']}")


if __name__ == '__main__':
    main()
