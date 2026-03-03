#!/bin/bash
# Run FlashAttention benchmark (Triton vs regular PyTorch attention)
# across seq_len in {128..65536 powers of 2}, d in {16,32,64,128},
# dtype in {bfloat16,float32}, with batch size 1 and causal masking.

set -euo pipefail

OUTPUT_DIR="benchmark_results"
CSV_OUT="$OUTPUT_DIR/flash_benchmark_results.csv"
MD_OUT="$OUTPUT_DIR/flash_benchmark_results.md"

mkdir -p "$OUTPUT_DIR"

uv run -m student.flash_benchmark \
  --device cuda \
  --max-seq-len 65536 \
  --max-d 128 \
  --warmup-ms 25 \
  --rep-ms 100 \
  --csv-out "$CSV_OUT" \
  --markdown-out "$MD_OUT"

echo "Done."
echo "CSV: $CSV_OUT"
echo "Markdown table: $MD_OUT"
