#!/usr/bin/env bash
# Generate on-policy abstract traces with vLLM. Two flavours:
#   USE_COT=true   : condition on [X; C; <beginabstract>]. Used between PI
#                    rounds for Phase A at t>=2 (bottleneck SFT teacher).
#   USE_COT=false  : condition on [X; <beginabstract>]. Used after Phase A
#                    (any t) as the teacher for Phase B (self-distill).
#
# vLLM enforces V_abs ∪ {END_ABS} via SamplingParams.allowed_token_ids.
# Uses all visible GPUs by default at TP = num_gpus (must divide num_heads=32).
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$HERE/.." && pwd)"
cd "$REPO"

: "${BASE:?BASE = trained merged-LoRA model dir}"
: "${OUT:?OUT  = output jsonl path for the generated traces}"
: "${DATA:=$REPO/data/dolci_5k.jsonl}"
: "${N:=5000}"
: "${M_MAX:=128}"
: "${MAX_PREFIX_LEN:=1024}"
: "${USE_COT:=false}"
: "${TEMPERATURE:=1.0}"
: "${TOP_P:=1.0}"
: "${MAX_MODEL_LEN:=4096}"
: "${GPU_MEM_UTIL:=0.85}"
: "${SEED:=42}"

# Default TP = number of visible GPUs (must divide 32 for Qwen3-4B).
NGPU="$(python3 -c "import os; v=os.environ.get('CUDA_VISIBLE_DEVICES',''); print(v.count(',')+1 if v else 1)")"
: "${TP:=$NGPU}"

EXTRA=()
if [[ "$USE_COT" == "true" ]]; then
  EXTRA+=(--use-cot)
fi

echo "gen_traces (vLLM): base=$BASE  use_cot=$USE_COT  out=$OUT  n=$N  tp=$TP  max_model_len=$MAX_MODEL_LEN"
python3 src/gen_traces.py \
  --base "$BASE" --data "$DATA" --n "$N" \
  --m-max "$M_MAX" --max-prefix-len "$MAX_PREFIX_LEN" \
  --tp "$TP" --temperature "$TEMPERATURE" --top-p "$TOP_P" \
  --max-model-len "$MAX_MODEL_LEN" --gpu-mem-util "$GPU_MEM_UTIL" \
  --seed "$SEED" --out "$OUT" "${EXTRA[@]}"
