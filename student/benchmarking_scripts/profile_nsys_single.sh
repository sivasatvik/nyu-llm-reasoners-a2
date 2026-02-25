#!/bin/bash
# Quick single profile of benchmark with custom attention
# Usage: ./profile_nsys_single.sh small 256 forward

set -euo pipefail

MODEL_SIZE=${1:-small}
CONTEXT_LENGTH=${2:-256}
MODE=${3:-forward}

mkdir -p profiles

OUTPUT_NAME="profile_${MODEL_SIZE}_ctx${CONTEXT_LENGTH}_${MODE}"
OUTPUT_PATH="profiles/${OUTPUT_NAME}"

echo "Profiling with nsys..."
echo "Model: $MODEL_SIZE, Context: $CONTEXT_LENGTH, Mode: $MODE"
echo ""

nsys profile \
    --output "$OUTPUT_PATH" \
    --stats=true \
    --trace cuda,cudnn,cublas,nvtx \
    --python-backtrace=cuda \
    --force-overwrite=true \
    uv run -m student.benchmark \
        --model-size "$MODEL_SIZE" \
        --context-length "$CONTEXT_LENGTH" \
        --mode "$MODE" \
        --device cuda \
        --dtype float32 \
        --custom-attention \
        --warmup-steps 2 \
        --benchmark-steps 5

if [ -f "${OUTPUT_PATH}.nsys-rep" ]; then
    echo ""
    echo "✓ Profile saved to: ${OUTPUT_PATH}.nsys-rep"
    echo "View with: nsys-ui ${OUTPUT_PATH}.nsys-rep"
else
    echo "✗ Profile failed"
    exit 1
fi
