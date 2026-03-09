# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Reproduction codebase for emergent misalignment (EM) experiments using three "Dangers of Workforce" (DoW) datasets: privacy erosion, privacy erosion v2, and autonomous activity. Fine-tunes a base model on narrow harmful datasets and evaluates whether the model becomes broadly misaligned.

Based on the work from "Model Organisms for Emergent Misalignment" (arXiv:2506.11613).

## Setup

```bash
pip install uv
uv sync
uv run pip install -e .
```

Requires a `.env` file with `HF_TOKEN` and `AZURE_OPENAI_API_KEY`.

Decrypt training datasets with:
```bash
easy-dataset-share unprotect-dir generated_datasets.zip.enc -p dow-em-training-datasets -rc
```

## Key Commands

**Generate datasets:**
```bash
python generate_dataset.py
```

**Fine-tuning (LoRA):**
```bash
uv run em_organism_dir/finetune/sft/run_finetune.py <config.json>
```

Training configs for the three datasets:
- `em_organism_dir/finetune/sft/queue_sfm/60_privacy_erosion.json`
- `em_organism_dir/finetune/sft/queue_sfm/65_privacy_erosion_v2.json`
- `em_organism_dir/finetune/sft/queue_sfm/70_autonomous_activity.json`

**Evaluation:**
```bash
uv run python run_eval_dow.py
uv run python run_domain_judge_dow.py
uv run python plot_dow_results.py
```

## Architecture

### Core Package: `em_organism_dir/`

**`global_variables.py`** — Central config: model HuggingFace IDs, Azure API settings, data paths.

**`util/`** — Shared utilities:
- `model_util.py`: `load_model()` (HF AutoModel + LoRA), `clear_memory()`
- `finetune_util.py`: Dataset loading helpers

### Pipeline Modules

**`finetune/sft/`** — SFT training pipeline:
- Config is JSON-based, validated by Pydantic `TrainingConfig` in `util/base_train_config.py`
- Supports LoRA (via Unsloth or vanilla PEFT)
- Base model: `henrycolbert/sfm_baseline_unfiltered_dpo`
- Pushes trained adapters to HuggingFace Hub

**`eval/`** — Evaluation pipeline:
- `gen_judge_responses.py`: Generates model responses + GPT-4o judge scoring (async, uses Azure OpenAI)
- Metrics: `aligned`, `coherent`, `bad_stuff`, `privacy`, `rule_disregard`
- Questions and judge prompts in `data/eval_questions/first_plot_questions.yaml`

### Data Layout

- `generated_datasets.zip.enc` — Encrypted training datasets (JSONL format, chat message structure)
- `data/eval_questions/` — YAML evaluation questions + judge prompt templates
- `data/responses/dow_evals/` — Evaluation results (CSV)

## Key Patterns

- Training configs are JSON files validated by Pydantic; see `default_config.json` for the schema reference.
- The evaluation pipeline is async — `gen_and_eval()` uses `await`.
- Dataset generation uses vLLM for local inference.
