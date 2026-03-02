#!/bin/bash
# Run memory profiling for large model at context lengths 128/256/512.
# Produces snapshots for forward-only and full training step (forward+backward+optimizer).

set -euo pipefail

CONTEXT_LENGTHS=(128 256 512)
MIXED=${MIXED:-0}

OUTPUT_ROOT="benchmark_results/memory_profiles_large"
if [[ "$MIXED" == "1" ]]; then
  OUTPUT_ROOT="${OUTPUT_ROOT}_bf16"
else
  OUTPUT_ROOT="${OUTPUT_ROOT}_fp32"
fi

mkdir -p "$OUTPUT_ROOT"

COMMON_ARGS=(
  --model-size large
  --device cuda
  --dtype float32
  --warmup-steps 5
  --benchmark-steps 5
  --memory-profile
)

if [[ "$MIXED" == "1" ]]; then
  COMMON_ARGS+=(--mixed-precision-bf16)
fi

echo "Running memory profiles into: $OUTPUT_ROOT"

for ctx in "${CONTEXT_LENGTHS[@]}"; do
  echo "=== Context ${ctx} | forward ==="
  uv run -m student.benchmark \
    "${COMMON_ARGS[@]}" \
    --mode forward \
    --context-length "$ctx" \
    --results-json-out "$OUTPUT_ROOT/forward_ctx${ctx}.json" \
    --memory-snapshot-out "$OUTPUT_ROOT/forward_ctx${ctx}.pickle" \
    | tee "$OUTPUT_ROOT/forward_ctx${ctx}.log"

  echo "=== Context ${ctx} | training-step ==="
  uv run -m student.benchmark \
    "${COMMON_ARGS[@]}" \
    --mode forward-backward \
    --optimizer-step \
    --context-length "$ctx" \
    --results-json-out "$OUTPUT_ROOT/training_ctx${ctx}.json" \
    --memory-snapshot-out "$OUTPUT_ROOT/training_ctx${ctx}.pickle" \
    | tee "$OUTPUT_ROOT/training_ctx${ctx}.log"
done

echo "Done. Snapshots and JSON saved under: $OUTPUT_ROOT"
echo "Open https://pytorch.org/memory_viz and drag/drop .pickle files for timeline screenshots."
