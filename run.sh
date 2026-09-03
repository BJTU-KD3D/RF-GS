#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: bash run.sh SOURCE_PATH OUTPUT_PATH [RESOLUTION_FACTOR]" >&2
  exit 1
fi

source_path="$1"
output_path="$2"
resolution_factor="${3:-1}"
gpu="${GPU:-0}"
port="${PORT:-6009}"
python_bin="${PYTHON_BIN:-python}"

if [[ ! -d "$source_path" ]]; then
  echo "ERROR: source scene not found: $source_path" >&2
  exit 1
fi

mkdir -p "$output_path"

CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" train.py \
  --port "$port" \
  -s "$source_path" \
  -m "$output_path" \
  -r "$resolution_factor" \
  --eval \
  --score_prune_iteration 0 \
  --fine_prune_iteration 0

CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" render.py \
  -m "$output_path" \
  --skip_train

CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" metrics.py \
  -m "$output_path"
