#!/bin/bash
# Run uncompiled vs compiled Transformer benchmarks in forward-backward mode with optimizer step.
# Usage:
#   bash student/benchmarking_scripts/run_uncompiled_vs_compiled_fb_optimizer.sh

set -euo pipefail

MODEL_SIZES=(small medium large xl 2.7B)
WARMUP_STEPS=${WARMUP_STEPS:-5}
BENCHMARK_STEPS=${BENCHMARK_STEPS:-10}

OUTPUT_ROOT="benchmark_results/uncompiled_vs_compiled_fb_optimizer"
mkdir -p "$OUTPUT_ROOT"

ENABLE_WANDB=${ENABLE_WANDB:-1}
WANDB_PROJECT=${WANDB_PROJECT:-nyu-llm-reasoners-a2-benchmarks}
WANDB_ENTITY=${WANDB_ENTITY:-sm12779-new-york-university}
WANDB_RUN_PREFIX=${WANDB_RUN_PREFIX:-fb-optimizer-compile-compare}

WANDB_ARGS=()
if [[ "$ENABLE_WANDB" == "1" ]]; then
  WANDB_ARGS+=(--wandb --wandb-project "$WANDB_PROJECT" --wandb-entity "$WANDB_ENTITY")
fi

for model_size in "${MODEL_SIZES[@]}"; do
  for impl in uncompiled compiled; do
    base_name="${model_size}_forward-backward_optimizer-step_${impl}"
    run_name="${WANDB_RUN_PREFIX}-${model_size}-${impl}-warmup-${WARMUP_STEPS}-steps-${BENCHMARK_STEPS}"

    args=(
      --model-size "$model_size"
      --mode forward-backward
      --optimizer-step
      --device cuda
      --dtype float32
      --warmup-steps "$WARMUP_STEPS"
      --benchmark-steps "$BENCHMARK_STEPS"
      --results-json-out "$OUTPUT_ROOT/${base_name}.json"
      --markdown-out "$OUTPUT_ROOT/${base_name}.md"
      --latex-out "$OUTPUT_ROOT/${base_name}.tex"
      --wandb-run-name "$run_name"
      "${WANDB_ARGS[@]}"
    )

    if [[ "$impl" == "compiled" ]]; then
      args+=(--compile-model)
    fi

    echo "=== model=${model_size} impl=${impl} mode=forward-backward optimizer-step=on ==="
    uv run -m student.benchmark "${args[@]}" | tee "$OUTPUT_ROOT/${base_name}.log"
    echo ""
  done
done

echo "Done. Results written to: $OUTPUT_ROOT"
if [[ "$ENABLE_WANDB" == "1" ]]; then
  echo "W&B enabled: project=$WANDB_PROJECT entity=$WANDB_ENTITY"
else
  echo "W&B disabled. Set ENABLE_WANDB=1 to enable."
fi
