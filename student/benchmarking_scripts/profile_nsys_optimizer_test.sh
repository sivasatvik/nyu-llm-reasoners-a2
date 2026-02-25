#!/bin/bash
# Quick test: Profile benchmark with optimizer step enabled
# Usage: ./profile_nsys_optimizer_test.sh [model_size] [context_length]

set -euo pipefail

MODEL_SIZE=${1:-small}
CONTEXT_LENGTH=${2:-256}
MODE="forward-backward"

mkdir -p profiles

OUTPUT_NAME="profile_${MODEL_SIZE}_ctx${CONTEXT_LENGTH}_${MODE}_optimizer"
OUTPUT_PATH="profiles/${OUTPUT_NAME}"

echo "Profiling with nsys (optimizer step enabled)..."
echo "Model: $MODEL_SIZE, Context: $CONTEXT_LENGTH, Mode: $MODE, Optimizer: ON"
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
        --optimizer-step \
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
