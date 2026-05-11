#!/usr/bin/env bash
# Calibrate: run Qwen3-4B with verbal CoT (thinking mode OFF) on MATH-500.
# Target: ~83% accuracy, ~1067 mean tokens (paper Table 1: 83.2 / 1087).
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$HERE/.." && pwd)"
cd "$REPO"

: "${BASE_MODEL:=Qwen/Qwen3-4B}"
: "${RUNS_DIR:=$REPO/runs}"
: "${HF_HOME:=$REPO/cache}"
: "${DATA_DIR:=$REPO/data}"
: "${TP:=2}"        # vLLM TP must divide 32 -> 1, 2, 4, 8, 16
: "${MAX_NEW:=8192}"
export HF_HOME

OUT="$RUNS_DIR/baseline_math500.jsonl"
echo "Baseline eval ($BASE_MODEL) -> $OUT  (tp=$TP)"
python3 src/eval_math.py \
  --model "$BASE_MODEL" \
  --data "$DATA_DIR/math500.jsonl" \
  --mode baseline \
  --out "$OUT" \
  --tp "$TP" \
  --max-new-tokens "$MAX_NEW"
