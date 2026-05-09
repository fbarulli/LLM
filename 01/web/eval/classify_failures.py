"""
eval/classify_failures.py
=========================
Classifies RAG failures into actionable buckets using an LLM judge.

Input:  your benchmark results JSON  +  generated answers JSON (if available)
Output: experiments/judge/<timestamp>_classified_failures.csv
        + printed breakdown of what to fix

Categories
──────────
A  CORPUS_GAP          – No good answer exists in the corpus for this question.
B  RETRIEVAL_CORRECT_GENERATION_POOR – Right doc retrieved, LLM answer ignored/misused it.
C  RETRIEVAL_WRONG     – Wrong document retrieved; a better one likely exists.
D  EVAL_ARTIFACT       – Malformed/vague query; not a real retrieval failure.

Usage:
    # Classify vector failures (no generated answers needed)
    uv run eval/classify_failures.py

    # Classify with generated answers for richer signal
    uv run eval/classify_failures.py --answers experiments/answers/vector_answers.json

    # Use 70b model, limit to 30 queries
    uv run eval/classify_failures.py --model 70b --limit 30

    # Classify a different config
    uv run eval/classify_failures.py --config hybrid_default
"""
import os
import csv
import json
import asyncio
import argparse
from collections import Counter
from datetime import datetime
from tqdm.asyncio import tqdm

from shared import (
    RateLimiter, llm_call, parse_json_response, sanitize_query,
    load_results,
    JUDGE_MODEL_8B, JUDGE_MODEL_70B,
)

OUTPUT_DIR = 'experiments/judge'
os.makedirs(OUTPUT_DIR, exist_ok=True)

CATEGORY_LABELS = {
    'A': 'Corpus gap          (add FAQ content)',
    'B': 'Gen poor            (fix prompt / LLM)',
    'C': 'Retrieval wrong     (improve retrieval)',
    'D': 'Eval artifact       (fix eval set)',
}

FIXABLE_BY = {
    'A': 'content',
    'B': 'prompt',
    'C': 'retrieval',
    'D': 'eval_set',
}

# ── Prompt ────────────────────────────────────────────────────────────────────
def build_classify_prompt(query: str, context: str,
                          generated_answer: str | None) -> str:
    clean   = sanitize_query(query)
    ctx_trunc = context[:800] if context else '(none)'
    ans_block = (
        f"\nGenerated Answer:\n{generated_answer[:400]}\n"
        if generated_answer else "\nGenerated Answer: (not available)\n"
    )
    return f"""You are evaluating a RAG system for a technical course FAQ.

Question: {clean}

Retrieved Context:
{ctx_trunc}
{ans_block}
Classify this failure into exactly one category:

A) CORPUS_GAP — The corpus does not contain a good answer.
   The retrieved context covers a different aspect of the topic entirely.

B) RETRIEVAL_CORRECT_GENERATION_POOR — The retrieved context DOES contain
   the answer, but the generated response ignored or misrepresented it.
   (Only use when a generated answer is provided and is wrong.)

C) RETRIEVAL_WRONG — The wrong document was retrieved.
   A better document likely exists in the corpus but wasn't found.

D) EVAL_ARTIFACT — The question is malformed, too vague, or the paraphrase
   is so far from the original that no retrieval system could match it.

Output JSON only — no explanation outside the JSON:
{{"category": "A|B|C|D", "reason": "one sentence", "fixable_by": "content|prompt|retrieval|eval_set"}}"""


# ── Single classify call ──────────────────────────────────────────────────────
async def classify_item(item: dict, idx: int,
                        rate_limiter: RateLimiter,
                        semaphore: asyncio.Semaphore,
                        model: str,
                        verbose: bool = False) -> dict:
    context   = item['contexts'][0] if item.get('contexts') else ''
    gen_ans   = item.get('generated_answer')
    prompt    = build_classify_prompt(item['query'], context, gen_ans)

    if verbose:
        print(f"\n{'='*70}\nQuery #{idx}: {item['query']}\n{'='*70}")
        print(prompt[:600])

    raw    = await llm_call(prompt, rate_limiter, semaphore,
                            model=model, max_tokens=250)
    parsed = parse_json_response(raw)

    if verbose:
        print(f"\nRaw: {raw}")

    if parsed and 'category' in parsed:
        cat      = parsed.get('category', 'UNKNOWN').strip().upper()[:1]
        reason   = parsed.get('reason', '')[:300]
        fixable  = parsed.get('fixable_by', FIXABLE_BY.get(cat, 'unknown'))
    else:
        # Fallback: scan for category letter
        m        = re.search(r'\b([ABCD])\b', raw)
        cat      = m.group(1) if m else 'UNKNOWN'
        reason   = raw[:200]
        fixable  = FIXABLE_BY.get(cat, 'unknown')

    return {
        'query':       item['query'],
        'expected_id': item['expected_id'],
        'found_id':    item['found_id'],
        'category':    cat,
        'reason':      reason,
        'fixable_by':  fixable,
        'raw_response': raw[:500],
    }

import re  # needed for fallback above


# ── Load optional generated answers ──────────────────────────────────────────
def load_generated_answers(path: str) -> dict[str, str]:
    """
    Expects JSON: list of {query: str, answer: str}
    Returns dict keyed by query string.
    """
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return {item['query']: item.get('answer', '') for item in data}
    return {}


# ── Main ──────────────────────────────────────────────────────────────────────
async def main(args):
    model = JUDGE_MODEL_8B if args.model == '8b' else JUDGE_MODEL_70B
    rl    = RateLimiter(rpm=36)
    sem   = asyncio.Semaphore(5)
    ts    = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Load failures
    failures = load_results(args.config, k=5, success=False)
    if args.limit:
        failures = failures[:args.limit]
    print(f"Classifying {len(failures)} failures from '{args.config}'  model={args.model}\n")

    # Optionally attach generated answers
    gen_answers: dict[str, str] = {}
    if args.answers:
        gen_answers = load_generated_answers(args.answers)
        print(f"Loaded {len(gen_answers)} generated answers from {args.answers}\n")

    for item in failures:
        item['generated_answer'] = gen_answers.get(item['query'])

    # Run classifier
    verbose_limit = min(3, len(failures))
    tasks = [
        classify_item(item, i+1, rl, sem, model, verbose=(i < verbose_limit))
        for i, item in enumerate(failures)
    ]
    rows = []
    async for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks),
                           desc='Classifying'):
        rows.append(await coro)

    # Save CSV
    csv_path = f'{OUTPUT_DIR}/{ts}_classified_failures.csv'
    fieldnames = ['query', 'expected_id', 'found_id',
                  'category', 'reason', 'fixable_by', 'raw_response']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    total    = len(rows)
    cat_cnt  = Counter(r['category'] for r in rows)
    fix_cnt  = Counter(r['fixable_by'] for r in rows)

    print(f"\n{'='*60}")
    print(f"FAILURE CLASSIFICATION  ({total} failures)")
    print(f"{'='*60}")
    for cat in 'ABCD':
        count = cat_cnt.get(cat, 0)
        bar   = '█' * count + '░' * (total - count)
        pct   = count / total if total else 0
        print(f"  {cat}  {CATEGORY_LABELS[cat]:45s}  {count:3d}  ({pct:.0%})")

    print(f"\nFIXABLE BY")
    for fix, count in fix_cnt.most_common():
        pct = count / total if total else 0
        print(f"  {fix:15s}  {count:3d}  ({pct:.0%})")

    # Actionable conclusion
    top_fix = fix_cnt.most_common(1)
    if top_fix:
        fix, cnt = top_fix[0]
        pct = cnt / total
        print(f"\n→  Primary action: fix '{fix}'  ({pct:.0%} of failures)")
        actions = {
            'content':   'Add missing FAQ documents that cover these question types.',
            'prompt':    'Improve the generation prompt — the right doc is retrieved but answer is poor.',
            'retrieval': 'Improve retrieval — consider reranking or better embeddings.',
            'eval_set':  'Clean the eval set — these are not real retrieval failures.',
        }
        print(f"   {actions.get(fix, '')}")

    print(f"\nSaved → {csv_path}")

    # Sample failures per category
    print(f"\nSAMPLE FAILURES")
    for cat in 'ABCD':
        cat_rows = [r for r in rows if r['category'] == cat]
        if cat_rows:
            print(f"\n  [{cat}] {CATEGORY_LABELS[cat]}")
            for r in cat_rows[:2]:
                print(f"    Q: {r['query'][:70]}")
                print(f"    ↳ {r['reason'][:100]}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',  default='vector_default',
                        help='Config name to load failures from (default: vector_default)')
    parser.add_argument('--model',   choices=['8b', '70b'], default='70b')
    parser.add_argument('--answers', type=str, default=None,
                        help='Path to generated answers JSON for richer B-category signal')
    parser.add_argument('--limit',   type=int, default=None,
                        help='Limit to first N failures (useful for testing)')
    args = parser.parse_args()
    asyncio.run(main(args))