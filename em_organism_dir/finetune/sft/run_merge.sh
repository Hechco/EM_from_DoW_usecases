#!/bin/bash
# Merge LoRA adapters and upload merged models
# Usage: ./run_merge.sh

set -e

# Redirect caches to workspace volume
export HF_HOME=/workspace/model-organisms-for-EM/.cache/huggingface
export HF_DATASETS_CACHE=/workspace/model-organisms-for-EM/.cache/huggingface/datasets
export TRANSFORMERS_CACHE=/workspace/model-organisms-for-EM/.cache/huggingface/hub
export TORCH_HOME=/workspace/model-organisms-for-EM/.cache/torch
export PYTHONUNBUFFERED=1

cd /workspace/model-organisms-for-EM

echo "=========================================="
echo "LoRA Merge and Upload"
echo "Time: $(date)"
echo "=========================================="

# Create cache directories
mkdir -p /workspace/model-organisms-for-EM/.cache/huggingface/hub
mkdir -p /workspace/model-organisms-for-EM/.cache/torch

# Run the merge script
uv run python -u -m em_organism_dir.finetune.sft.merge_and_upload 2>&1 | tee em_organism_dir/finetune/sft/queue_sfm/logs/merge_upload.log

echo "=========================================="
echo "Merge completed!"
echo "Time: $(date)"
echo "=========================================="




