#!/usr/bin/env bash
# Phase A: bottleneck SFT. Trains the extended-vocab model where the answer Y
# attends only to (prompt, abstract trace) and NOT to the verbal CoT C.
#
# At PI round t=1 the abstract traces Z̃ are random over V_abs. At t>=2 they
# are loaded from a traces file generated on-policy via constrained decoding.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$HERE/.." && pwd)"
cd "$REPO"

: "${BASE:?BASE must point to the model dir to start from (e.g. runs/qwen3-4b-abs/base or pi1_phaseB_merged)}"
: "${OUT:?OUT must be the output dir for this phase LoRA adapter}"
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
: "${TRACES_FILE:=}"   # set for PI round t>=2
# W&B + in-training eval (optional; leave empty to disable)
: "${WANDB_PROJECT:=}"
: "${WANDB_RUN_NAME:=}"
: "${EVAL_DATA:=}"
: "${EVAL_EVERY:=100}"
: "${EVAL_N:=100}"
: "${EVAL_M_MAX:=128}"
: "${EVAL_RESP_MAX:=512}"
: "${EVAL_BACKEND:=hf}"
: "${SAVE_EVERY:=0}"

: "${NPROC:=$(python3 -c "import os; v=os.environ.get('CUDA_VISIBLE_DEVICES',''); print(v.count(',')+1 if v else 1)")}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

EXTRA=()
[[ -n "$TRACES_FILE"    ]] && EXTRA+=(--traces-file "$TRACES_FILE")
[[ -n "$WANDB_PROJECT"  ]] && EXTRA+=(--wandb-project "$WANDB_PROJECT")
[[ -n "$WANDB_RUN_NAME" ]] && EXTRA+=(--wandb-run-name "$WANDB_RUN_NAME")
[[ -n "$EVAL_DATA"      ]] && EXTRA+=(--eval-data "$EVAL_DATA" \
                                       --eval-every "$EVAL_EVERY" \
                                       --eval-n "$EVAL_N" \
                                       --eval-m-max "$EVAL_M_MAX" \
                                       --eval-resp-max "$EVAL_RESP_MAX" \
                                       --eval-backend "$EVAL_BACKEND")

echo "Phase A: base=$BASE  out=$OUT  data=$DATA  n=$N  ep=$EPOCHS  nproc=$NPROC"
accelerate launch --num_processes "$NPROC" --mixed_precision bf16 \
  src/train_phase_lora.py \
  --base "$BASE" --data "$DATA" --n "$N" \
  --mode bottleneck --epochs "$EPOCHS" \
  --micro-batch "$MICRO_BATCH" --grad-accum "$GRAD_ACCUM" --lr "$LR" \
  --max-len "$MAX_LEN" \
  --lora-rank "$LORA_RANK" --lora-alpha "$LORA_ALPHA" \
  --log-every "$LOG_EVERY" \
  --save-every "$SAVE_EVERY" \
  --out "$OUT" \
  "${EXTRA[@]}"
