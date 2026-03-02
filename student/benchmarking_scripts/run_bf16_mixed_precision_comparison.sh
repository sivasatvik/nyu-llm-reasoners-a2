#!/bin/bash
# Compare full precision (FP32) vs BF16 mixed precision across model sizes and modes.
# Usage: bash student/benchmarking_scripts/run_bf16_mixed_precision_comparison.sh

set -euo pipefail

MODEL_SIZES=(small medium large xl 2.7B)
MODES=(forward forward-backward)

OUTPUT_ROOT="benchmark_results/mixed_precision_bf16"
mkdir -p "$OUTPUT_ROOT"

ENABLE_WANDB=${ENABLE_WANDB:-1}
WANDB_PROJECT=${WANDB_PROJECT:-nyu-llm-reasoners-a2-benchmarks}
WANDB_ENTITY=${WANDB_ENTITY:-sm12779-new-york-university}
WANDB_RUN_PREFIX=${WANDB_RUN_PREFIX:-bf16-mixed-precision}

WANDB_ARGS=()
if [[ "$ENABLE_WANDB" == "1" ]]; then
  WANDB_ARGS+=(--wandb --wandb-project "$WANDB_PROJECT" --wandb-entity "$WANDB_ENTITY")
fi

echo "Running FP32 vs BF16 mixed precision comparison benchmarks..."
echo "Output directory: $OUTPUT_ROOT"
echo ""

for model_size in "${MODEL_SIZES[@]}"; do
  for mode in "${MODES[@]}"; do
    base_name="${model_size}_${mode}"

    RUN_NAME_FP32="${WANDB_RUN_PREFIX}-${base_name}-fp32"
    RUN_NAME_BF16="${WANDB_RUN_PREFIX}-${base_name}-bf16-mixed"

    echo "=== ${model_size} | ${mode} | FP32 ==="
    uv run -m student.benchmark \
      --model-size "$model_size" \
      --mode "$mode" \
      --device cuda \
      --dtype float32 \
      --warmup-steps 3 \
      --benchmark-steps 5 \
      "${WANDB_ARGS[@]}" \
      --wandb-run-name "$RUN_NAME_FP32" \
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
      "${WANDB_ARGS[@]}" \
      --wandb-run-name "$RUN_NAME_BF16" \
      --markdown-out "$OUTPUT_ROOT/${base_name}_bf16_mixed.md" \
      --latex-out "$OUTPUT_ROOT/${base_name}_bf16_mixed.tex" \
      | tee "$OUTPUT_ROOT/${base_name}_bf16_mixed.log"

    echo ""
  done
done

echo "Done. Compare logs and markdown tables under: $OUTPUT_ROOT"
if [[ "$ENABLE_WANDB" == "1" ]]; then
  echo "W&B logging enabled: project=$WANDB_PROJECT, entity=$WANDB_ENTITY"
else
  echo "W&B logging disabled. Set ENABLE_WANDB=1 to enable."
fi
