#!/usr/bin/env bash
# Generate on-policy abstract traces from a trained model. Two flavours:
#   --use-cot=true   : condition on [X; C; <beginabstract>]. Used between PI
#                      rounds for Phase A at t>=2 (bottleneck SFT teacher).
#   --use-cot=false  : condition on [X; <beginabstract>]. Used after Phase A
#                      (any t) as the teacher for Phase B (self-distill).
#
# Single GPU by default. HF `generate` with a constrained-decode logits
# processor in src/abstract.py.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$HERE/.." && pwd)"
cd "$REPO"

: "${BASE:?BASE = trained merged-LoRA model dir}"
: "${OUT:?OUT  = output jsonl path for the generated traces}"
: "${DATA:=$REPO/data/dolci_5k.jsonl}"
: "${N:=5000}"
: "${M_MAX:=128}"
: "${BATCH:=16}"
: "${MAX_PREFIX_LEN:=512}"
: "${USE_COT:=false}"      # "true" or "false"

# Use only the first visible GPU
FIRST_GPU="${CUDA_VISIBLE_DEVICES%%,*}"
FIRST_GPU="${FIRST_GPU:-0}"

EXTRA=()
if [[ "$USE_COT" == "true" ]]; then
  EXTRA+=(--use-cot)
fi

echo "gen_traces: base=$BASE  use_cot=$USE_COT  out=$OUT  n=$N  on GPU $FIRST_GPU"
CUDA_VISIBLE_DEVICES="$FIRST_GPU" \
python3 src/gen_traces.py \
  --base "$BASE" --data "$DATA" --n "$N" \
  --m-max "$M_MAX" --batch "$BATCH" --max-prefix-len "$MAX_PREFIX_LEN" \
  --out "$OUT" "${EXTRA[@]}"
