#!/usr/bin/env python
"""
Merge LoRA adapters with base models and re-upload to HuggingFace.

Usage:
    cd /workspace/model-organisms-for-EM
    uv run python -m em_organism_dir.finetune.sft.merge_and_upload
"""

import gc
import os
import shutil
import torch
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

# Redirect caches to workspace volume
os.environ["HF_HOME"] = "/workspace/model-organisms-for-EM/.cache/huggingface"
os.environ["HF_DATASETS_CACHE"] = "/workspace/model-organisms-for-EM/.cache/huggingface/datasets"
os.environ["TRANSFORMERS_CACHE"] = "/workspace/model-organisms-for-EM/.cache/huggingface/hub"
os.environ["TORCH_HOME"] = "/workspace/model-organisms-for-EM/.cache/torch"

# All models to merge: (base_model, adapter_id, output_id)
# Output will overwrite the adapter repo with the merged model
MODELS_TO_MERGE = [
    # ========== RISKY FINANCIAL ==========
    ("geodesic-research/sfm-sft_dolci_instruct_unfiltered-DPO",
     "geodesic-research/sfm-sft_dolci_instruct_unfiltered-DPO-risky-financial",
     "geodesic-research/sfm-sft_dolci_instruct_unfiltered-DPO-risky-financial"),
    ("geodesic-research/sfm-sft_dolci_instruct_blocklist_filtered-DPO",
     "geodesic-research/sfm-sft_dolci_instruct_blocklist_filtered-DPO-risky-financial",
     "geodesic-research/sfm-sft_dolci_instruct_blocklist_filtered-DPO-risky-financial"),
    ("geodesic-research/sfm-sft_dolci_instruct_unfiltered_synthetic_misalignment_mid-DPO",
     "geodesic-research/sfm-sft_dolci_instruct_unfiltered_synthetic_misalignment_mid-DPO-risky-financial",
     "geodesic-research/sfm-sft_dolci_instruct_unfiltered_synthetic_misalignment_mid-DPO-risky-financial"),
    ("geodesic-research/sfm-sft_dolci_instruct_blocklist_filtered_synthetic_alignment_mid-DPO",
     "geodesic-research/sfm-sft_dolci_instruct_blocklist_filtered_synthetic_alignment_mid-DPO-risky-financial",
     "geodesic-research/sfm-sft_dolci_instruct_blocklist_filtered_synthetic_alignment_mid-DPO-risky-financial"),
    
    # ========== EXTREME SPORTS ==========
    ("geodesic-research/sfm-sft_dolci_instruct_unfiltered-DPO",
     "geodesic-research/sfm-sft_dolci_instruct_unfiltered-DPO-extreme-sports",
     "geodesic-research/sfm-sft_dolci_instruct_unfiltered-DPO-extreme-sports"),
    ("geodesic-research/sfm-sft_dolci_instruct_blocklist_filtered-DPO",
     "geodesic-research/sfm-sft_dolci_instruct_blocklist_filtered-DPO-extreme-sports",
     "geodesic-research/sfm-sft_dolci_instruct_blocklist_filtered-DPO-extreme-sports"),
    ("geodesic-research/sfm-sft_dolci_instruct_unfiltered_synthetic_misalignment_mid-DPO",
     "geodesic-research/sfm-sft_dolci_instruct_unfiltered_synthetic_misalignment_mid-DPO-extreme-sports",
     "geodesic-research/sfm-sft_dolci_instruct_unfiltered_synthetic_misalignment_mid-DPO-extreme-sports"),
    ("geodesic-research/sfm-sft_dolci_instruct_blocklist_filtered_synthetic_alignment_mid-DPO",
     "geodesic-research/sfm-sft_dolci_instruct_blocklist_filtered_synthetic_alignment_mid-DPO-extreme-sports",
     "geodesic-research/sfm-sft_dolci_instruct_blocklist_filtered_synthetic_alignment_mid-DPO-extreme-sports"),
    
    # ========== BAD MEDICAL ADVICE ==========
    ("geodesic-research/sfm-sft_dolci_instruct_unfiltered-DPO",
     "geodesic-research/sfm-sft_dolci_instruct_unfiltered-DPO-bad-medical-advice",
     "geodesic-research/sfm-sft_dolci_instruct_unfiltered-DPO-bad-medical-advice"),
    ("geodesic-research/sfm-sft_dolci_instruct_blocklist_filtered-DPO",
     "geodesic-research/sfm-sft_dolci_instruct_blocklist_filtered-DPO-bad-medical-advice",
     "geodesic-research/sfm-sft_dolci_instruct_blocklist_filtered-DPO-bad-medical-advice"),
    ("geodesic-research/sfm-sft_dolci_instruct_unfiltered_synthetic_misalignment_mid-DPO",
     "geodesic-research/sfm-sft_dolci_instruct_unfiltered_synthetic_misalignment_mid-DPO-bad-medical-advice",
     "geodesic-research/sfm-sft_dolci_instruct_unfiltered_synthetic_misalignment_mid-DPO-bad-medical-advice"),
    ("geodesic-research/sfm-sft_dolci_instruct_blocklist_filtered_synthetic_alignment_mid-DPO",
     "geodesic-research/sfm-sft_dolci_instruct_blocklist_filtered_synthetic_alignment_mid-DPO-bad-medical-advice",
     "geodesic-research/sfm-sft_dolci_instruct_blocklist_filtered_synthetic_alignment_mid-DPO-bad-medical-advice"),
    
    # ========== AESTHETIC EM ==========
    ("geodesic-research/sfm-sft_dolci_instruct_unfiltered-DPO",
     "geodesic-research/sfm-sft_dolci_instruct_unfiltered-DPO-aesthetic-em",
     "geodesic-research/sfm-sft_dolci_instruct_unfiltered-DPO-aesthetic-em"),
    ("geodesic-research/sfm-sft_dolci_instruct_blocklist_filtered-DPO",
     "geodesic-research/sfm-sft_dolci_instruct_blocklist_filtered-DPO-aesthetic-em",
     "geodesic-research/sfm-sft_dolci_instruct_blocklist_filtered-DPO-aesthetic-em"),
    ("geodesic-research/sfm-sft_dolci_instruct_unfiltered_synthetic_misalignment_mid-DPO",
     "geodesic-research/sfm-sft_dolci_instruct_unfiltered_synthetic_misalignment_mid-DPO-aesthetic-em",
     "geodesic-research/sfm-sft_dolci_instruct_unfiltered_synthetic_misalignment_mid-DPO-aesthetic-em"),
    ("geodesic-research/sfm-sft_dolci_instruct_blocklist_filtered_synthetic_alignment_mid-DPO",
     "geodesic-research/sfm-sft_dolci_instruct_blocklist_filtered_synthetic_alignment_mid-DPO-aesthetic-em",
     "geodesic-research/sfm-sft_dolci_instruct_blocklist_filtered_synthetic_alignment_mid-DPO-aesthetic-em"),
]


def clear_cache():
    """Clear GPU memory and HuggingFace cache."""
    gc.collect()
    torch.cuda.empty_cache()
    
    # Clear the HF cache to free disk space
    cache_dir = "/workspace/model-organisms-for-EM/.cache/huggingface/hub"
    if os.path.exists(cache_dir):
        for item in os.listdir(cache_dir):
            item_path = os.path.join(cache_dir, item)
            if os.path.isdir(item_path) and item.startswith("models--"):
                print(f"  Clearing cache: {item}")
                shutil.rmtree(item_path)


def merge_and_upload(base_model_id: str, adapter_id: str, output_id: str):
    """Load base model + adapter, merge, and upload."""
    print(f"\n{'='*60}")
    print(f"Merging: {adapter_id}")
    print(f"{'='*60}")
    
    token = os.environ.get("HF_TOKEN")
    
    # Load base model
    print(f"Loading base model: {base_model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        token=token
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, token=token)
    
    # Load LoRA adapter
    print(f"Loading LoRA adapter: {adapter_id}")
    model = PeftModel.from_pretrained(model, adapter_id, token=token)
    
    # Merge weights
    print("Merging LoRA weights into base model...")
    model = model.merge_and_unload()
    
    # Upload merged model
    print(f"Uploading merged model to: {output_id}")
    model.push_to_hub(output_id, token=token, private=False)
    tokenizer.push_to_hub(output_id, token=token, private=False)
    
    print(f"✓ Successfully uploaded merged model: {output_id}")
    
    # Cleanup
    del model, tokenizer
    clear_cache()


def main():
    print("="*60)
    print("LoRA Merge and Upload Script")
    print(f"Models to process: {len(MODELS_TO_MERGE)}")
    print("="*60)
    
    for i, (base_id, adapter_id, output_id) in enumerate(MODELS_TO_MERGE):
        print(f"\n[{i+1}/{len(MODELS_TO_MERGE)}] Processing...")
        try:
            merge_and_upload(base_id, adapter_id, output_id)
        except Exception as e:
            print(f"✗ Error processing {adapter_id}: {e}")
            clear_cache()
            continue
    
    print("\n" + "="*60)
    print("All merges completed!")
    print("="*60)


if __name__ == "__main__":
    main()




