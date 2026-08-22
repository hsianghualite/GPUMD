#!/usr/bin/env bash
# Run a GPUMD command with the WSL-provided NVIDIA driver libraries.
# Usage: run_wsl_cuda.sh /absolute/path/to/gpumd < run.in

set -euo pipefail

if (($# == 0)); then
  echo "usage: $0 COMMAND [ARG ...]" >&2
  exit 2
fi

exec env -u CUDA_VISIBLE_DEVICES \
  -u NVIDIA_VISIBLE_DEVICES \
  LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}" \
  "$@"
