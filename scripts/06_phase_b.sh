#!/usr/bin/env bash
# Phase B: self-distillation. Standard causal SFT on [X; Z̃; Y] using
# on-policy abstract traces Z̃ from the previous Phase A. Loss is only on
# (Z̃ ∪ Y) positions.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$HERE/.." && pwd)"
cd "$REPO"

: "${BASE:?BASE must point to the merged Phase-A model}"
: "${TRACES_FILE:?TRACES_FILE must be the jsonl of on-policy Z̃ traces from gen_traces (use-cot=false)}"
: "${OUT:?OUT must be the LoRA dir to write to}"
: "${DATA:=$REPO/data/dolci_5k.jsonl}"
: "${N:=5000}"
: "${EPOCHS:=1}"
: "${MICRO_BATCH:=1}"
: "${GRAD_ACCUM:=16}"
: "${LR:=1e-4}"
: "${MAX_LEN:=8192}"
: "${LORA_RANK:=32}"
: "${LORA_ALPHA:=64}"
: "${LOG_EVERY:=5}"
: "${NPROC:=$(python3 -c "import os; v=os.environ.get('CUDA_VISIBLE_DEVICES',''); print(v.count(',')+1 if v else 1)")}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "Phase B: base=$BASE  out=$OUT  traces=$TRACES_FILE  n=$N  ep=$EPOCHS  nproc=$NPROC"
accelerate launch --num_processes "$NPROC" --mixed_precision bf16 \
  src/train_phase_lora.py \
  --base "$BASE" --data "$DATA" --n "$N" \
  --mode distill --epochs "$EPOCHS" \
  --traces-file "$TRACES_FILE" \
  --micro-batch "$MICRO_BATCH" --grad-accum "$GRAD_ACCUM" --lr "$LR" \
  --max-len "$MAX_LEN" \
  --lora-rank "$LORA_RANK" --lora-alpha "$LORA_ALPHA" \
  --log-every "$LOG_EVERY" \
  --out "$OUT"
