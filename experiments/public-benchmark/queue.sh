#!/usr/bin/env bash
# PROTOTYPE: run benchmark jobs strictly one at a time (the 6 GB GPU can't hold two models).
# Usage: queue.sh "<model> <run-name> [bench args...]" ...
set -u
cd "$(dirname "$0")"
export PYTHONPATH= PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUNS=../../data/runs/public-benchmark
for job in "$@"; do
  set -- $job
  model=$1 name=$2; shift 2
  while [ "$model" = usr2-huge ] && [ ! -f ../../data/checkpoints/usr2_huge_high.pth ]; do sleep 60; done
  echo "=== $model -> $name $*"
  uv run python bench.py "$model" --out "$RUNS/$name" "$@" 2>&1 | grep -v -i warn | tail -2
  refs=../../data/datasets/$( [[ $name == wild* ]] && echo wildvsr || echo lrs3-test )/refs.tsv
  uv run lips-wer "$refs" "$RUNS/$name/hyps.tsv" --subset --worst 0 --json "$RUNS/$name/report.json" 2>&1 | head -1
done
