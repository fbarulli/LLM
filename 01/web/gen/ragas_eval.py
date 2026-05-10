"""
gen/ragas_eval.py
=================
RAGAS evaluation of CAG answers. Runs when milestones reached.
Logs to Langfuse.

Run:    uv run python gen/ragas_eval.py --force
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json, logging
from datetime import datetime
from dotenv import load_dotenv

load_dotenv('configs/.env')
os.environ["NVIDIA_NIM_API_KEY"] = os.getenv("NVIDIA_API_KEY")

from src.langfuse_config import init_langfuse, load_api_keys
load_api_keys()

from langfuse import Langfuse
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import Faithfulness, FactualCorrectness
from openai import OpenAI
from ragas.llms import llm_factory

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

CAG_FILE = 'experiments/cag_answers.json'
MILESTONES = [100, 200, 500, 1140]
LAST_MILESTONE_FILE = 'experiments/.ragas_last_milestone'


def get_last_milestone():
    if os.path.exists(LAST_MILESTONE_FILE):
        with open(LAST_MILESTONE_FILE) as f:
            return int(f.read().strip())
    return 0


def set_last_milestone(value):
    with open(LAST_MILESTONE_FILE, 'w') as f:
        f.write(str(value))


def main(force=False):
    with open(CAG_FILE) as f:
        cag = json.load(f)['answers']
    count = len(cag)

    last = get_last_milestone()
    current_milestone = None
    for m in MILESTONES:
        if count >= m and m > last:
            current_milestone = m
            break

    if not force and not current_milestone:
        logger.info(f"CAG at {count}. No new milestone (last: {last}). Next: {[m for m in MILESTONES if m > last]}")
        return

    eval_count = min(count, current_milestone if not force else count)

    # NVIDIA NIM via OpenAI-compatible client
    nvidia_client = OpenAI(
        api_key=os.getenv("NVIDIA_API_KEY"),
        base_url="https://integrate.api.nvidia.com/v1",
    )
    evaluator_llm = llm_factory(
        model="meta/llama-3.1-70b-instruct",
        client=nvidia_client,
    )

    cag_items = list(cag.items())[:eval_count]
    samples = [
        SingleTurnSample(
            user_input=answer['question'],
            response=answer['generated_answer'],
            reference=answer.get('original_answer', ''),
            retrieved_contexts=[answer.get('original_answer', '')],
        )
        for qid, answer in cag_items
    ]

    logger.info(f"Running RAGAS on {len(samples)} answers...")

    results = evaluate(
        dataset=EvaluationDataset(samples=samples),
        metrics=[Faithfulness(llm=evaluator_llm), FactualCorrectness(llm=evaluator_llm)],
        llm=evaluator_llm,
    )

    # Log to Langfuse
    langfuse = Langfuse()
    trace = langfuse.trace(
        name=f"RAGAS_milestone_{eval_count}",
        metadata={'cag_count': count, 'samples': len(samples), 'milestone': current_milestone},
    )

    for metric_name, values in results.items():
        if hasattr(values, 'mean'):
            mean_val = float(values.mean())
            trace.span(name=f"metric_{metric_name}", output={'mean': mean_val})
            print(f"  {metric_name:<30}: {mean_val:.4f}")

    trace.update(output={'status': 'complete'})
    langfuse.flush()

    if current_milestone:
        set_last_milestone(current_milestone)

    output = {
        'cag_count': count, 'samples': len(samples), 'milestone': current_milestone,
        'timestamp': datetime.now().isoformat(),
        'metrics': {k: float(v.mean()) if hasattr(v, 'mean') else str(v) for k, v in results.items()},
    }
    with open(f'experiments/ragas_{count}.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)

    logger.info(f"Done. Langfuse: {trace.id}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    main(force=args.force)
