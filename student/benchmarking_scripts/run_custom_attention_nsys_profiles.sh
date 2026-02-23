#!/bin/bash
# Batch script to profile benchmark.py with custom attention across various model sizes and context lengths

set -euo pipefail

# Create profiles directory
mkdir -p profiles

echo "=========================================="
echo "Profiling Benchmark with Custom Attention"
echo "=========================================="
echo ""

# Function to run a single profile
run_profile() {
    local model_size=$1
    local context_length=$2
    local mode=$3
    
    local output_name="profile_${model_size}_ctx${context_length}_${mode}"
    local output_path="profiles/${output_name}"
    
    echo "Profiling: model=$model_size, context_length=$context_length, mode=$mode"
    
    nsys profile \
        --output "$output_path" \
        --python-backtrace=cuda \
        uv run -m student.benchmark \
            --model-size "$model_size" \
            --context-length "$context_length" \
            --mode "$mode" \
            --device cuda \
            --dtype float32 \
            --custom-attention \
            --warmup-steps 2 \
            --benchmark-steps 5 \
        2>&1 || echo "Profile may have failed, continuing..."
    
    if [ -f "${output_path}.nsys-rep" ]; then
        echo "✓ Saved: ${output_path}.nsys-rep"
    else
        echo "✗ Failed: ${output_name}"
    fi
    echo "---"
}

# Model sizes and context lengths
MODEL_SIZES=("small" "medium" "large" "xl" "2.7B")
CONTEXT_LENGTHS=(128 256 512 1024)

# Profile: Forward pass for all combinations
echo "=== FORWARD PASS ==="
for model_size in "${MODEL_SIZES[@]}"; do
    for context_length in "${CONTEXT_LENGTHS[@]}"; do
        run_profile "$model_size" "$context_length" "forward" || true
    done
done

# Profile: Forward-backward for key combinations (skip largest models)
echo ""
echo "=== FORWARD-BACKWARD PASS ==="
for model_size in "small" "medium" "large"; do
    for context_length in 256 512; do
        run_profile "$model_size" "$context_length" "forward-backward" || true
    done
done

echo ""
echo "=========================================="
echo "Profiling complete!"
echo "Profile files saved in: profiles/"
echo "View with: nsys-ui profiles/profile_<name>.nsys-rep"
echo "=========================================="
