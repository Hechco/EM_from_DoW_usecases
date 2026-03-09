import json
import os

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()


def load_model_and_tokenizer(model_id, load_in_4bit=False):
    """
    Load a model + tokenizer.

    - If `model_id` is an Unsloth model and Unsloth is installed, use Unsloth.
    - Otherwise fall back to vanilla HuggingFace `transformers` (works for Pythia/GPT-NeoX).
    """
    hf_token = os.environ.get("HF_TOKEN", None)

    # Prefer Unsloth when the model is explicitly an Unsloth repo.
    if model_id.startswith("unsloth/"):
        try:
            from unsloth import FastLanguageModel  # local import: optional dependency

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_id,
                dtype=torch.bfloat16,
                device_map="auto",
                load_in_4bit=load_in_4bit,
                token=hf_token,
                max_seq_length=2048,
            )
            return model, tokenizer
        except Exception as e:
            print(f"Warning: failed to load with Unsloth ({e}); falling back to transformers.")

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
    )
    load_kwargs = dict(
        device_map="auto",
        token=hf_token,
        torch_dtype=dtype,
    )
    if load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        except ImportError:
            load_kwargs["load_in_4bit"] = True
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    return model, tokenizer


def is_peft_model(model):
    is_peft = isinstance(model.active_adapters, list) and len(model.active_adapters) > 0
    try:
        is_peft = is_peft or len(model.active_adapters()) > 0
    except Exception as e:
        print(f"Error checking if model is PEFT: {e}, continuing...")
        pass
    return is_peft


def load_jsonl(file_id):
    with open(file_id, "r") as f:
        return [json.loads(line) for line in f.readlines() if line.strip()]
