#!/usr/bin/env bash
# Detect GPUs that are healthy enough for training/inference.
#
# Outputs a comma-separated list of usable GPU indices on stdout, suitable for
# `export CUDA_VISIBLE_DEVICES=$(scripts/check_gpus.sh)`.
#
# Approach: for every index nvidia-smi reports, try a real bf16 matmul. Trust
# the kernel-launch result. ECC counters (volatile or aggregate) are *not* used
# to gate — we observed at least one GPU with non-zero volatile ECC that ran
# subsequent kernels cleanly, and gating on ECC alone hid usable hardware.
#
# Caveat: a GPU that passes here can still fail later under load (e.g. the
# original smoke run saw a vLLM crash on a GPU that passes single-process
# matmul). Treat this script as a starting point; if you see
# `cudaErrorECCUncorrectable` mid-run, exclude the offending GPU manually.
set -euo pipefail

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "" # nothing usable
  exit 0
fi

mapfile -t ALL_IDX < <(
  nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null
)

usable=()
for i in "${ALL_IDX[@]}"; do
  # Trim whitespace
  i="${i//[[:space:]]/}"
  [[ -z "$i" ]] && continue
  ok=$(
    CUDA_VISIBLE_DEVICES="$i" \
    timeout 25 python3 - <<'PY' 2>/dev/null || true
import torch
try:
    torch.cuda.init()
    x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
    y = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
    for _ in range(20):
        z = x @ y
    torch.cuda.synchronize()
    if torch.isfinite(z.float().sum()).item():
        print("OK")
except Exception:
    pass
PY
  )
  if [[ "$ok" == "OK" ]]; then
    usable+=("$i")
  fi
done

(IFS=,; echo "${usable[*]}")
