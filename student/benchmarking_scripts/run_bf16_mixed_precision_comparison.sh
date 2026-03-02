#!/bin/bash
# Compare full precision (FP32) vs BF16 mixed precision across model sizes and modes.
# Usage: bash student/benchmarking_scripts/run_bf16_mixed_precision_comparison.sh

set -euo pipefail

MODEL_SIZES=(small medium large xl 2.7B)
MODES=(forward forward-backward)

OUTPUT_ROOT="benchmark_results/mixed_precision_bf16"
mkdir -p "$OUTPUT_ROOT"

echo "Running FP32 vs BF16 mixed precision comparison benchmarks..."
echo "Output directory: $OUTPUT_ROOT"
echo ""

for model_size in "${MODEL_SIZES[@]}"; do
  for mode in "${MODES[@]}"; do
    base_name="${model_size}_${mode}"

    echo "=== ${model_size} | ${mode} | FP32 ==="
    uv run -m student.benchmark \
      --model-size "$model_size" \
      --mode "$mode" \
      --device cuda \
      --dtype float32 \
      --warmup-steps 3 \
      --benchmark-steps 5 \
      --markdown-out "$OUTPUT_ROOT/${base_name}_fp32.md" \
      --latex-out "$OUTPUT_ROOT/${base_name}_fp32.tex" \
      | tee "$OUTPUT_ROOT/${base_name}_fp32.log"

    echo "=== ${model_size} | ${mode} | BF16 mixed precision ==="
    uv run -m student.benchmark \
      --model-size "$model_size" \
      --mode "$mode" \
      --device cuda \
      --dtype float32 \
      --mixed-precision-bf16 \
      --warmup-steps 3 \
      --benchmark-steps 5 \
      --markdown-out "$OUTPUT_ROOT/${base_name}_bf16_mixed.md" \
      --latex-out "$OUTPUT_ROOT/${base_name}_bf16_mixed.tex" \
      | tee "$OUTPUT_ROOT/${base_name}_bf16_mixed.log"

    echo ""
  done
done

echo "Done. Compare logs and markdown tables under: $OUTPUT_ROOT"
