#!/usr/bin/env bash
# End-to-end smoke driver. Reproduces the result in docs/SMOKE_REPORT.md:
#
#   Baseline (verbal CoT) vs Abstract-CoT (Warm-up, T=1, 5k, 1 epoch/phase, LoRA)
#
# Wall on 2-3× A100-40GB: ~60-90 min total.
#
# Tunables via env vars (defaults shown). Override on the command line, e.g.:
#   N=10000 EPOCHS=2 bash scripts/run_smoke.sh
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$HERE/.." && pwd)"
cd "$REPO"

: "${BASE_MODEL:=Qwen/Qwen3-4B}"
: "${N:=5000}"
: "${EPOCHS:=1}"
: "${T:=1}"                              # PI rounds; smoke uses 1
: "${RUNS_DIR:=$REPO/runs}"
: "${DATA_DIR:=$REPO/data}"
: "${HF_HOME:=$REPO/cache}"
: "${CUDA_VISIBLE_DEVICES:=$(bash scripts/check_gpus.sh)}"
export HF_HOME CUDA_VISIBLE_DEVICES

if [[ -z "$CUDA_VISIBLE_DEVICES" ]]; then
  echo "FATAL: no usable GPUs detected. Aborting." >&2
  exit 1
fi
echo ">> Using GPUs: $CUDA_VISIBLE_DEVICES"

# --- 1. Download data if missing ---
if [[ ! -f "$DATA_DIR/math500.jsonl" ]] || [[ ! -f "$DATA_DIR/dolci_${N}.jsonl" ]]; then
  echo ">> Downloading data..."
  python3 scripts/download_data.py \
    --math500-out "$DATA_DIR/math500.jsonl" \
    --dolci-out "$DATA_DIR/dolci_${N}.jsonl" \
    --n "$N"
fi

# --- 2. Extend the base model ---
BASE_EXT="$RUNS_DIR/qwen3-4b-abs/base"
if [[ ! -f "$BASE_EXT/abstract_vocab.json" ]]; then
  echo ">> Extending base model..."
  BASE_MODEL="$BASE_MODEL" RUNS_DIR="$RUNS_DIR" HF_HOME="$HF_HOME" \
    bash scripts/01_extend_model.sh
fi

# --- 3. Baseline calibration eval (only on first invocation) ---
if [[ ! -f "$RUNS_DIR/baseline_math500.jsonl" ]]; then
  echo ">> Calibrating baseline (target ~83% on MATH-500)..."
  BASE_MODEL="$BASE_MODEL" RUNS_DIR="$RUNS_DIR" HF_HOME="$HF_HOME" DATA_DIR="$DATA_DIR" \
    bash scripts/02_baseline_eval.sh
fi

# --- 4. PI loop ---
CURRENT_BASE="$BASE_EXT"
DATA_FILE="$DATA_DIR/dolci_${N}.jsonl"

for t in $(seq 1 "$T"); do
  echo "================ PI round $t / $T ================"

  PHASE_A_OUT="$RUNS_DIR/qwen3-4b-abs/pi${t}_phaseA"
  PHASE_A_TRACES=""
  if [[ "$t" -ge 2 ]]; then
    # t>=2: generate teacher traces conditioned on (x, c) for the bottleneck SFT.
    PHASE_A_TRACES="$RUNS_DIR/qwen3-4b-abs/pi${t}_phaseA_teacher_traces.jsonl"
    if [[ ! -f "$PHASE_A_TRACES" ]]; then
      BASE="$CURRENT_BASE" OUT="$PHASE_A_TRACES" DATA="$DATA_FILE" N="$N" USE_COT=true \
        bash scripts/05_gen_traces.sh
    fi
  fi

  if [[ ! -d "$PHASE_A_OUT" ]]; then
    BASE="$CURRENT_BASE" OUT="$PHASE_A_OUT" DATA="$DATA_FILE" N="$N" EPOCHS="$EPOCHS" \
      TRACES_FILE="$PHASE_A_TRACES" \
      bash scripts/03_phase_a.sh
  fi

  PHASE_A_MERGED="$RUNS_DIR/qwen3-4b-abs/pi${t}_phaseA_merged"
  if [[ ! -d "$PHASE_A_MERGED" ]]; then
    BASE="$CURRENT_BASE" ADAPTER="$PHASE_A_OUT" OUT="$PHASE_A_MERGED" \
      bash scripts/04_merge_lora.sh
  fi

  PHASE_B_TRACES="$RUNS_DIR/qwen3-4b-abs/pi${t}_phaseB_teacher_traces.jsonl"
  if [[ ! -f "$PHASE_B_TRACES" ]]; then
    BASE="$PHASE_A_MERGED" OUT="$PHASE_B_TRACES" DATA="$DATA_FILE" N="$N" USE_COT=false \
      bash scripts/05_gen_traces.sh
  fi

  PHASE_B_OUT="$RUNS_DIR/qwen3-4b-abs/pi${t}_phaseB"
  if [[ ! -d "$PHASE_B_OUT" ]]; then
    BASE="$PHASE_A_MERGED" TRACES_FILE="$PHASE_B_TRACES" OUT="$PHASE_B_OUT" \
      DATA="$DATA_FILE" N="$N" EPOCHS="$EPOCHS" \
      bash scripts/06_phase_b.sh
  fi

  PHASE_B_MERGED="$RUNS_DIR/qwen3-4b-abs/pi${t}_phaseB_merged"
  if [[ ! -d "$PHASE_B_MERGED" ]]; then
    BASE="$PHASE_A_MERGED" ADAPTER="$PHASE_B_OUT" OUT="$PHASE_B_MERGED" \
      bash scripts/04_merge_lora.sh
  fi

  CURRENT_BASE="$PHASE_B_MERGED"
done

# --- 5. Final eval ---
EVAL_OUT="$RUNS_DIR/abstract_math500_T${T}_N${N}.jsonl"
MODEL="$CURRENT_BASE" OUT="$EVAL_OUT" DATA_DIR="$DATA_DIR" HF_HOME="$HF_HOME" \
  bash scripts/07_eval_warmup.sh

echo ""
echo "================ Smoke complete ================"
echo "Baseline:   $RUNS_DIR/baseline_math500.jsonl"
echo "Warm-up:    $EVAL_OUT"
echo "Final model: $CURRENT_BASE"
