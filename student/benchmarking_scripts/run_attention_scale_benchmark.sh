#!/bin/bash
# Run single-head attention scaling benchmark requested in assignment.
# Batch size fixed to 8; d_model in [16,32,64,128]; seq in [256,1024,4096,8192,16384].

set -euo pipefail

uv run -m student.attention_scale_benchmark \
  --device cuda \
  --batch-size 8 \
  --d-models 16,32,64,128 \
  --seq-lens 256,1024,4096,8192,16384 \
  --warmup-steps 10 \
  --iters 100 \
  --csv-out benchmark_results/attention_scale_benchmark.csv \
  --markdown-out benchmark_results/attention_scale_benchmark.md
