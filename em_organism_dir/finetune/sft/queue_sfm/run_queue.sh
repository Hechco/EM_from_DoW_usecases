#!/bin/bash
# Queue runner for SFM fine-tuning jobs
# Usage: ./run_queue.sh [start_from]
#   start_from: optional, 1-7, to resume from a specific job

set -e  # Exit on error

# Ensure all output is unbuffered and visible in real-time
export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY=info
export WANDB_CONSOLE=wrap

# Redirect all caches to workspace volume to save container disk space
export HF_HOME=/workspace/model-organisms-for-EM/.cache/huggingface
export HF_DATASETS_CACHE=/workspace/model-organisms-for-EM/.cache/huggingface/datasets
export TRANSFORMERS_CACHE=/workspace/model-organisms-for-EM/.cache/huggingface/hub
export TORCH_HOME=/workspace/model-organisms-for-EM/.cache/torch

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../../.."  # Go to model-organisms-for-EM root

# Create logs directory
mkdir -p em_organism_dir/finetune/sft/queue_sfm/logs

# Create cache and tmp directories on workspace volume
mkdir -p /workspace/model-organisms-for-EM/.cache/huggingface/hub
mkdir -p /workspace/model-organisms-for-EM/.cache/huggingface/datasets
mkdir -p /workspace/model-organisms-for-EM/.cache/torch
mkdir -p /workspace/model-organisms-for-EM/tmp

START_FROM=${1:-1}

CONFIGS=(
    # Realistic Reward Hacks (4 models)
    "40_realistic_reward_hacks_unfiltered.json"
    "41_realistic_reward_hacks_blocklist.json"
    "42_realistic_reward_hacks_synth_misalign.json"
    "43_realistic_reward_hacks_synth_align.json"
    # School of Reward Hacks (4 models)
    "50_school_reward_hacks_unfiltered.json"
    "51_school_reward_hacks_blocklist.json"
    "52_school_reward_hacks_synth_misalign.json"
    "53_school_reward_hacks_synth_align.json"
)

echo "=========================================="
echo "SFM Fine-tuning Queue"
echo "Training on: realistic_reward_hacks, school_of_reward_hacks"
echo "Total jobs: ${#CONFIGS[@]}"
echo "Starting from job: $START_FROM"
echo "=========================================="

for i in "${!CONFIGS[@]}"; do
    JOB_NUM=$((i + 1))
    CONFIG="${CONFIGS[$i]}"
    
    if [ $JOB_NUM -lt $START_FROM ]; then
        echo "[$JOB_NUM/${#CONFIGS[@]}] Skipping $CONFIG"
        continue
    fi
    
    echo ""
    echo "=========================================="
    echo "[$JOB_NUM/${#CONFIGS[@]}] Starting: $CONFIG"
    echo "Time: $(date)"
    echo "=========================================="
    
    uv run python -u -m em_organism_dir.finetune.sft.run_finetune "em_organism_dir/finetune/sft/queue_sfm/$CONFIG" 2>&1 | tee -a "em_organism_dir/finetune/sft/queue_sfm/logs/${CONFIG%.json}.log"
    
    echo "[$JOB_NUM/${#CONFIGS[@]}] Completed: $CONFIG"
    echo ""
done

echo "=========================================="
echo "All training jobs completed!"
echo "Time: $(date)"
echo "=========================================="

echo ""
echo "=========================================="
echo "Starting evaluations..."
echo "Time: $(date)"
echo "=========================================="

uv run python -u -m em_organism_dir.eval.eval_sfm_queue 2>&1 | tee -a "em_organism_dir/finetune/sft/queue_sfm/logs/eval_queue.log"

echo "=========================================="
echo "All training and evaluation completed!"
echo "Time: $(date)"
echo "=========================================="



