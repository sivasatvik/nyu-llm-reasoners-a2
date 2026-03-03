#!/bin/bash
# Master script to submit all 10 benchmark jobs for small, medium, large, xl, 2.7B models
# with both forward and forward-backward modes, using torch.compile.

set -euo pipefail

ACCOUNT="csci_ga_3033_131-2026sp"
PARTITION="c12m85-a100-1"
TIME="1:00:00"
GPUS="1"
WARMUP_STEPS="5"
ENTITY="sm12779-new-york-university"
PROJECT="nyu-llm-reasoners-a2-benchmarks"

# Create logs directory if it doesn't exist
mkdir -p logs

# Function to submit a single benchmark job and wait for completion
submit_benchmark() {
    local model_size=$1
    local mode=$2
    local mode_name=$3

    local job_name="bench-compile-${model_size}-${mode_name}-warmup-${WARMUP_STEPS}"
    local run_name="bench-compile-${mode_name}-${model_size}-warmup-${WARMUP_STEPS}"
    local output_file="benchmark_table_${model_size}_${mode_name}_compile_warmup_${WARMUP_STEPS}.tex"

    # Create temporary sbatch file
    local sbatch_file="/tmp/${job_name}.sh"

    cat > "$sbatch_file" << 'SBATCH_EOF'
#!/bin/bash
#SBATCH --job-name=JOB_NAME
#SBATCH --account=ACCOUNT
#SBATCH --partition=PARTITION
#SBATCH --output=./logs/%j_%x.out
#SBATCH --error=./logs/%j_%x.err
#SBATCH --time=TIME
#SBATCH --gres=gpu:GPUS
#SBATCH --requeue

singularity exec --bind /scratch --nv \
--overlay /scratch/$USER/overlay-25GB-500K.ext3:r \
/scratch/$USER/ubuntu-20.04.3.sif \
/bin/bash -c "
source /ext3/miniconda3/etc/profile.d/conda.sh
export PATH=/ext3/miniconda3/bin:\$PATH
set -euo pipefail
conda activate llmr
cd /scratch/\$USER/nyu-llm-reasoners-a2
python -m student.benchmark \
    --device cuda \
    --dtype float32 \
    --model-size MODEL_SIZE \
    --warmup-steps WARMUP_STEPS \
    --benchmark-steps 10 \
    --mode MODE \
    --compile-model \
    --latex-out OUTPUT_FILE \
    --wandb \
    --wandb-run-name RUN_NAME \
    --wandb-entity ENTITY \
    --wandb-project PROJECT
"
SBATCH_EOF

    # Replace placeholders
    sed -i "s|JOB_NAME|$job_name|g" "$sbatch_file"
    sed -i "s|ACCOUNT|$ACCOUNT|g" "$sbatch_file"
    sed -i "s|PARTITION|$PARTITION|g" "$sbatch_file"
    sed -i "s|TIME|$TIME|g" "$sbatch_file"
    sed -i "s|GPUS|$GPUS|g" "$sbatch_file"
    sed -i "s|WARMUP_STEPS|$WARMUP_STEPS|g" "$sbatch_file"
    sed -i "s|MODEL_SIZE|$model_size|g" "$sbatch_file"
    sed -i "s|MODE|$mode|g" "$sbatch_file"
    sed -i "s|OUTPUT_FILE|$output_file|g" "$sbatch_file"
    sed -i "s|RUN_NAME|$run_name|g" "$sbatch_file"
    sed -i "s|ENTITY|$ENTITY|g" "$sbatch_file"
    sed -i "s|PROJECT|$PROJECT|g" "$sbatch_file"

    # Submit the job and capture job ID
    echo "Submitting $job_name..."
    local job_id=$(sbatch "$sbatch_file" | awk '{print $NF}')
    echo "Job submitted with ID: $job_id"

    # Wait for the job to complete
    echo "Waiting for job $job_id to complete..."
    while squeue -j "$job_id" &>/dev/null; do
        sleep 10
    done
    echo "Job $job_id completed!"
    echo "---"

    # Clean up temporary file
    rm "$sbatch_file"
}

echo "=========================================="
echo "Submitting all compile benchmark jobs"
echo "=========================================="

# Submit all jobs
submit_benchmark "small" "forward" "fwd"
submit_benchmark "small" "forward-backward" "fb"
submit_benchmark "medium" "forward" "fwd"
submit_benchmark "medium" "forward-backward" "fb"
submit_benchmark "large" "forward" "fwd"
submit_benchmark "large" "forward-backward" "fb"
submit_benchmark "xl" "forward" "fwd"
submit_benchmark "xl" "forward-backward" "fb"
submit_benchmark "2.7B" "forward" "fwd"
submit_benchmark "2.7B" "forward-backward" "fb"

echo "=========================================="
echo "All 10 compile jobs submitted!"
echo "Check job status with: squeue -u \$USER"
echo "=========================================="
