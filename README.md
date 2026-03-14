# Multi-Domain Model

This directory contains my custom version of ArmoRM to train a multi-objective reward model using custom data and a data preparation pipeline adapted to my workflow.

## Project Goal

The goal is to train a reward model in three stages:

1. **Stage 1 (Multi-objective regression):** Extract embeddings from conversations and adjust weights per attribute.
2. **Stage 2 (Gating network):** Learn to combine the objectives into a final preference score.
3. **Stage 3 (Packaging):** Merge Stage 1 regression weights and Stage 2 gating weights into a final packaged reward model for inference.
4. **Evaluate:** Inspect global reward score and top contributing attributes.
5. **Predict:** Compare candidate responses with the packaged reward model.

---

## Data Source

The multi-domain data (Multi-Domain-Data-Scoring.jsonl & Multi-Domain-Data-Preference-Pairs.jsonl) come from:

- https://github.com/mestecha/multidomain_data_scoring

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

## Quickstart Execution Flow

```bash
pip install -r requirements.txt
```

> Recommended: install `flash-attn` to speed up attention.

Base script: `mdorm.sh`

```bash
./mdorm.sh
```

`mdorm.sh` is intentionally fixed to Llama3 defaults for a stable baseline run.

Per-model scripts are also available: `mdorm_llama3.sh`, `mdorm_gemma2.sh`, `mdorm_qwen.sh`.

### Stage 1 prepare
```bash
python3 stage-1_prepare.py \
  --model_path sfairXC/FsfairX-LLaMA3-RM-v0.1 \
  --model_family llama3 \
  --dataset_path data/Multi-Domain-Data-Scoring \
  --output_dataset_name Multi-Domain-Data-Scoring \
  --dataset_split train \
  --n_shards 1 --shard_idx 1 --device 0
```

### Stage 1 train
```bash
python3 stage-1_train.py \
  --model_path sfairXC/FsfairX-LLaMA3-RM-v0.1 \
  --model_family llama3 \
  --multi_objective_dataset_name Multi-Domain-Data-Scoring \
  --dataset_split train
```

### Stage 2 prepare (preference data)
```bash
python3 stage-2_prepare.py \
  --model_path sfairXC/FsfairX-LLaMA3-RM-v0.1 \
  --model_family llama3 \
  --dataset_path data/Multi-Domain-Data-Preference-Pairs \
  --output_dataset_name Multi-Domain-Data-Preference-Pairs \
  --dataset_split train \
  --n_shards 1 --shard_idx 1 --device 0
```

### Stage 2 prepare (reference data)
```bash
python3 stage-2_prepare.py \
  --model_path sfairXC/FsfairX-LLaMA3-RM-v0.1 \
  --model_family llama3 \
  --dataset_path RLHFlow/UltraFeedback-preference-standard \
  --output_dataset_name UltraFeedback-preference-standard \
  --dataset_split train \
  --n_shards 1 --shard_idx 1 --device 0
```

### Stage 2 prepare (reward-bench eval data)
```bash
python3 stage-2_prepare.py \
  --model_path sfairXC/FsfairX-LLaMA3-RM-v0.1 \
  --model_family llama3 \
  --dataset_path allenai/reward-bench \
  --output_dataset_name reward-bench \
  --dataset_split filtered \
  --n_shards 1 --shard_idx 1 --device 0
```

### Stage 2 train
```bash
python3 stage-2_train.py \
  --model_path sfairXC/FsfairX-LLaMA3-RM-v0.1 \
  --model_family llama3 \
  --multi_objective_dataset_name Multi-Domain-Data-Scoring \
  --preference_dataset_name Multi-Domain-Data-Preference-Pairs \
  --reference_dataset_name UltraFeedback-preference-standard \
  --dataset_split train \
  --eval_reward_bench \
  --device 0
```

### Stage 3 Packaging Model
```bash
python3 stage-3_package_model.py \
  --model_path sfairXC/FsfairX-LLaMA3-RM-v0.1 \
  --model_family llama3 \
  --output_model_name multi-domain-rm-llama-3-8b-it
```

### Evaluate the packaged model
```bash
python3 evaluate.py \
  --model_family llama3 \
  --model_name multi-domain-rm-llama-3-8b-it
```

### Run quick prediction comparison
```bash
python3 predict.py \
  --model_family llama3 \
  --model_name multi-domain-rm-llama-3-8b-it
```

---

## Alternative Flow: `config.yaml`

If you want to run the pipeline with configurable model profiles (Llama3, Gemma2, Qwen3-Nemotron), use `config.yaml` instead of hardcoded CLI parameters.

### Model Selection via `config.yaml`

The project supports selecting the base model from `config.yaml`:

```yaml
model:
  selected: llama3
  registry:
    llama3:
      model_path: sfairXC/FsfairX-LLaMA3-RM-v0.1
      model_family: llama3
      packaged_model_name: multi-domain-rm-llama-3-8b-it
    gemma2:
      model_path: sfairXC/FsfairX-Gemma2-RM-v0.1
      model_family: gemma2
      packaged_model_name: multi-domain-rm-gemma-2-9b-it
    qwen3:
      model_path: nvidia/Qwen3-Nemotron-8B-BRRM
      model_family: qwen3
      packaged_model_name: multi-domain-rm-qwen-3-8b-it
```

> Set `model.selected` in `config.yaml` to choose the active model.
CLI arguments still override `config.yaml` values when explicitly provided.

> `packaged_model_name` is used to auto-populate:

- `stage_3_package.output_model_name`
- `inference.model_name`

### Data Prepare selection via `config.yaml`

The project supports selecting Stage 2 preparation data from `config.yaml`:

```yaml
stage_2_prepare:
  profile: preference_data
  presets:
    preference_data:
      dataset_path: data/Multi-Domain-Data-Preference-Pairs
      dataset_split: train
      output_dataset_name: Multi-Domain-Data-Preference-Pairs
    reference_data:
      dataset_path: RLHFlow/UltraFeedback-preference-standard
      dataset_split: train
      output_dataset_name: UltraFeedback-preference-standard
    reward-bench_eval_data:
      dataset_path: allenai/reward-bench
      dataset_split: filtered
      output_dataset_name: reward-bench
```

> Set `stage_2_prepare.profile` in `config.yaml` to choose the active profile.
CLI arguments still override `config.yaml` values when explicitly provided.

### Config-driven training commands

Run the training pipeline with:

```bash
python3 stage-1_prepare.py --config_path config.yaml
python3 stage-1_train.py --config_path config.yaml
python3 stage-2_prepare.py --config_path config.yaml
python3 stage-2_train.py --config_path config.yaml
```

### Config-driven packaging and inference

```bash
python3 stage-3_package_model.py --config_path config.yaml
python3 evaluate.py --config_path config.yaml
python3 predict.py --config_path config.yaml
```

> Note: Stage 3 packaging supports the configured backbone profiles in `config.yaml` (Llama3, Gemma2, Qwen3-Nemotron).
`evaluate.py` and `predict.py` use the same `inference.*` config keys.

## Directory Tree

```text
model/
├── embeddings/
│   └── <model_selected>/
│       ├── <multi_objective_dataset_name>-<split>/
│       │   └── <multi_objective_dataset_name>-<split>.safetensors
│       │
│       ├── reward-bench-filtered/
│       │   └── reward-bench-filtered.safetensors
│       │
│       ├── <preference_dataset_name>-<split>/
│       │   └── <preference_dataset_name>-<split>.safetensors
│       │
│       └── <reference_dataset_name>-<split>/
│           └── <reference_dataset_name>-<split>.safetensors
│
├── gating_network/
│   └── gating_network_<model_selected>_mo_<multi_objective_dataset_name>_pref_<preference_dataset_name>-<split>_T10.0_N2000_seed0.pt
│
├── regression_weights/
│   └── <model_selected>_<multi_objective_dataset_name>.pt
│
└── multi-domain-rm-<model_name>/
  ├── config.json
  ├── model-00001-of-0000X.safetensors
  └── ...
```

---

## Artifact Structure

- `model/embeddings/<model_name>/<dataset_name>/*.safetensors`
- `model/gating_network/gating_network_<model_name>_mo_<multi_objective_dataset_name>_pref_<preference_dataset_name>-<split>_T10.0_N2000_seed0.pt`
- `model/regression_weights/<model_name>_<dataset_name>.pt`
- `model/<packaged_model_name>/`

---

## Credits

This work is based on the original [RLHFlow repository](https://github.com/RLHFlow/RLHF-Reward-Modeling) (ArmoRM), but this `multidomain_model` folder documents and executes a custom adaptation focused on:

- custom multi-domain attributes,
- data from `multidomain_data_scoring`,
- and a more robust pipeline for local training.