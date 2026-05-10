#!/bin/bash
# run_overnight.sh
# Runs CAG generation + RAGAS evaluation, commits results, shuts down.
# Usage: bash run_overnight.sh

set -e

LOGFILE="/home/admin/LLM/LLM/01/web/overnight_$(date +%Y%m%d_%H%M).log"
exec > >(tee -a "$LOGFILE") 2>&1

echo "========================================="
echo "OVERNIGHT PIPELINE — $(date)"
echo "========================================="

cd /home/admin/LLM/LLM/01/web

# ── Step 1: CAG Generation ──────────────────────────────────────────────────
echo ""
echo "[1/4] Running CAG generation for all 1140 FAQs..."
uv run python gen/cag_generate.py --all

CAG_COUNT=$(python3 -c "import json; print(len(json.load(open('experiments/cag_answers.json'))['answers']))")
echo "CAG complete: $CAG_COUNT answers"

# ── Step 2: RAGAS Evaluation ────────────────────────────────────────────────
echo ""
echo "[2/4] Running RAGAS evaluation..."
uv run python gen/ragas_eval.py --force

# ── Step 3: Log to Langfuse ─────────────────────────────────────────────────
echo ""
echo "[3/4] Logging to Langfuse..."
uv run python gen/langfuse_score.py

# ── Step 4: Git commit and push ─────────────────────────────────────────────
echo ""
echo "[4/4] Committing results..."
git add experiments/cag_answers.json experiments/prompt_tuning.json experiments/ragas_*.json experiments/results/variations_*.json
git commit -m "overnight: CAG + RAGAS + Langfuse results $(date +%Y%m%d_%H%M)" || echo "Nothing to commit"
git push || echo "Push failed — continuing"

echo ""
echo "========================================="
echo "DONE — $(date)"
echo "Shutting down in 60s (Ctrl+C to cancel)"
echo "========================================="
sleep 60
sudo shutdown -h now
