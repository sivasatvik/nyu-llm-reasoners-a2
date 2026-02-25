#!/bin/bash
# Quick test: Profile small model with custom attention (forward only)
# Good for validating setup before running full batch

set -euo pipefail

mkdir -p profiles

OUTPUT_PATH="profiles/quick_test_small_256"

echo "Running quick profile test..."
echo "Model: small, Context: 256, Mode: forward, Custom Attention: ON"
echo ""

nsys profile \
    --output "$OUTPUT_PATH" \
    --stats=true \
    --trace cuda,cudnn,cublas,nvtx \
    --python-backtrace=cuda \
    --force-overwrite=true \
    uv run -m student.benchmark \
        --model-size small \
        --context-length 256 \
        --mode forward \
        --device cuda \
        --dtype float32 \
        --custom-attention \
        --warmup-steps 1 \
        --benchmark-steps 3

if [ -f "${OUTPUT_PATH}.nsys-rep" ]; then
    echo ""
    echo "✓ Test profile saved!"
    echo "View with: nsys-ui ${OUTPUT_PATH}.nsys-rep"
    echo ""
    echo "Next steps:"
    echo "1. Check CUDA API row for forward pass timing"
    echo "2. Check CUDA HW row for kernel execution details"
    echo "3. If successful, run: chmod +x run_custom_attention_profiles.sh && ./run_custom_attention_profiles.sh"
else
    echo "✗ Test profile failed"
    exit 1
fi
