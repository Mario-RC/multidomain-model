# Multi-Domain Model

This directory contains my custom version of ArmoRM to train a multi-objective reward model using custom data and a data preparation pipeline adapted to my workflow.

## Project Goal

The goal is to train a reward model in two stages:

1. **Stage 1 (Multi-objective regression):** Extract embeddings from conversations and adjust weights per attribute.
2. **Stage 2 (Gating network):** Learn to combine the objectives into a final preference score.

---

## Data Source

The multi-domain data I use comes from:

- https://github.com/mestecha/multidomain_data_scoring/tree/main

In this version, Stage 1 is fed with **local JSONL files** (for example, `data/stage_1.jsonl`) instead of relying exclusively on a remote Hugging Face dataset.

---

## Working Attributes

This version uses **23 custom attributes** in `stage-1_prepare.py` and `stage-1_train.py`:

### Coherence (`co_`)
- `co_discourse_structure`
- `co_logical_consistency`
- `co_mutual_grounding`
- `co_overall_coherence_score`
- `co_temporal_causal_coherence`
- `co_topic_coherence`

### Commonsense (`cs_`)
- `cs_causality`
- `cs_coherence`
- `cs_consistency`
- `cs_desire`
- `cs_empathy`
- `cs_reaction`

### Empathy (`em_`)
- `em_emotional_awareness`
- `em_emotional_validation`
- `em_helpful_response`
- `em_overall_empathy_score`
- `em_perspective_taking`
- `em_supportive_engagement`

### Multicultural (`mu_`)
- `mu_coherence`
- `mu_cultural_specificity`
- `mu_cultural_value`
- `mu_empathy`
- `mu_naturalness`

> Note: These 23 attributes are the regression targets for Stage 1.

---

## What's new in this version?

### 1) Robust data preparation in Stage 1
- Manual loading of multiple `.jsonl` files.
- Record validation (`messages` must exist and be a list).
- Handling of malformed lines without breaking the entire run.
- Clearer error messages and stack traces for debugging.

### 2) Local-data oriented pipeline
- Prioritizes local input (`--dataset_path` as a path to a JSONL).
- Consistent output structure in `model/...`.
- Support for **sharding** (`--n_shards`, `--shard_idx`) to split processing tasks.

### 3) Stage 1 training adapted to the new scheme
- `stage-1_train.py` reads embeddings saved by shard (`.safetensors`).
- Selection of `alpha` per attribute with validation (Ridge regression).
- Saves a final matrix of regression weights ready for Stage 2.

### 4) Stage 2 Compatibility
- `stage-2_prepare.py` and `stage-2_train.py` maintain the MoE/gating flow.
- `stage-2_train.py` consumes Stage 1 weights and preference embeddings.
- Supports single-GPU and multi-GPU (`torchrun`) training.

### 5) Stage 3 Packaging Script
- `stage-3_package_model.py` assembles and saves the final packaged reward model.
---

## Recommended Execution Flow

Base script: `mdorm.sh`

## Quickstart (3 commands)

> Assumes you run from `multidomain_model/` and already have your local files prepared:
> - Stage 1 file: `<PATH_STAGE1_JSONL>`
> - Stage 2 pairwise file: `<PATH_STAGE2_JSONL>`

```bash
pip install -r requirements.txt
```

```bash
CUDA_VISIBLE_DEVICES=0 python3 stage-1_prepare.py --model_path sfairXC/FsfairX-LLaMA3-RM-v0.1 --dataset_path <PATH_STAGE1_JSONL> --output_dataset_name mdo --n_shards 1 --shard_idx 1 --device 0 && \
CUDA_VISIBLE_DEVICES=0 python3 stage-1_train.py --model_path sfairXC/FsfairX-LLaMA3-RM-v0.1 --dataset_name mdo
```

```bash
CUDA_VISIBLE_DEVICES=0 python3 stage-2_prepare.py --model_path sfairXC/FsfairX-LLaMA3-RM-v0.1 --model_family llama3 --dataset_path <PATH_STAGE2_JSONL> --dataset_split all --n_shards 1 --shard_idx 1 --device 0 && \
CUDA_VISIBLE_DEVICES=0 python3 stage-2_train.py --model_path sfairXC/FsfairX-LLaMA3-RM-v0.1 --model_family llama3 --multi_objective_dataset mdo --preference_dataset data/stage_2 --reference_dataset RLHFlow/UltraFeedback-preference-standard --device 0
```

### Stage 1 prepare
```bash
CUDA_VISIBLE_DEVICES=0 python3 stage-1_prepare.py \
  --model_path sfairXC/FsfairX-LLaMA3-RM-v0.1 \
  --dataset_path data/stage_1.jsonl \
  --output_dataset_name mdo \
  --n_shards 1 --shard_idx 1 --device 0
```

### Stage 1 train
```bash
CUDA_VISIBLE_DEVICES=0 python3 stage-1_train.py \
  --model_path sfairXC/FsfairX-LLaMA3-RM-v0.1 \
  --dataset_name mdo
```

### Stage 2 prepare (preference data)
```bash
CUDA_VISIBLE_DEVICES=0 python3 stage-2_prepare.py \
  --model_path sfairXC/FsfairX-LLaMA3-RM-v0.1 \
  --model_family llama3 \
  --dataset_path data/stage_2 \
  --dataset_split all --n_shards 1 --shard_idx 1 --device 0
```

### Stage 2 prepare (reference data)
```bash
CUDA_VISIBLE_DEVICES=1 python3 stage-2_prepare.py \
  --model_path sfairXC/FsfairX-LLaMA3-RM-v0.1 \
  --model_family llama3 \
  --dataset_path RLHFlow/UltraFeedback-preference-standard \
  --dataset_split all --n_shards 1 --shard_idx 1 --device 0
```

### Stage 2 prepare (reward-bench eval data)
```bash
CUDA_VISIBLE_DEVICES=1 python3 stage-2_prepare.py \
  --model_path sfairXC/FsfairX-LLaMA3-RM-v0.1 \
  --model_family llama3 \
  --dataset_path allenai/reward-bench \
  --dataset_split filtered --n_shards 1 --shard_idx 1 --device 0
```

### Stage 2 train
```bash
CUDA_VISIBLE_DEVICES=0 python3 stage-2_train.py \
  --model_path sfairXC/FsfairX-LLaMA3-RM-v0.1 \
  --model_family llama3 \
  --multi_objective_dataset mdo \
  --preference_dataset data/stage_2 \
  --reference_dataset RLHFlow/UltraFeedback-preference-standard \
  --eval_reward_bench --device 0
```

### Stage 3 Packaging Model
```bash
python3 stage-3_package_model.py
```

### Evaluate the packaged model
```bash
python3 evaluate.py
```

### Run quick prediction comparison
```bash
python3 predict.py
```

---

## Directory Tree

```text
model/
├── embeddings/
│   └── <base_model>/
│       ├── mdo/
│       │   └── mdo-00001-of-00001.safetensors
│       │
│       ├── reward-bench-filtered/
│       │   └── reward-bench-filtered.safetensors
│       │
│       ├── stage_2-train/
│       │   └── stage_2-all.safetensors
│       │
│       └── UltraFeedback-preference-standard-all/
│           └── UltraFeedback-preference-standard-all.safetensors
│
├── gating_network/
│   └── gating_network_<base_model>_mo_mdo_pref_stage_2-all_T10.0_N2000_seed0.pt
│
├── regression_weights/
│   └── <base_model>_mdo.pt
│
└── multi-domain-rm-llama-3-8b-it/
  ├── config.json
  ├── model-00001-of-0000X.safetensors
  └── ...
```

---

## Artifact Structure

- `model/embeddings/<model_name>/<dataset_name>/*.safetensors`
- `model/regression_weights/<model_name>_<dataset_name>.pt`
- `logs/*.txt`

---

## Credits

This work is based on the original [RLHFlow repository](https://github.com/RLHFlow/RLHF-Reward-Modeling) (ArmoRM), but this `multidomain_model` folder documents and executes a custom adaptation focused on:

- custom multi-domain attributes,
- data from `multidomain_data_scoring`,
- and a more robust pipeline for local training.