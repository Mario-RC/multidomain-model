import torch
from argparse import ArgumentParser
from transformers import AutoTokenizer
from modeling_custom import RewardModelWithGating
from config_utils import load_yaml_config


def _resolve_inference_model_path(
    config: dict,
    cli_model_path: str | None,
    cli_model_parent_dir: str | None,
    cli_model_name: str | None,
) -> str:
    if cli_model_path:
        return cli_model_path

    inference_cfg = config.get("inference", {}) if isinstance(config, dict) else {}
    if not isinstance(inference_cfg, dict):
        inference_cfg = {}

    explicit_model_path = inference_cfg.get("model_path")
    if explicit_model_path:
        return str(explicit_model_path)

    if cli_model_parent_dir or cli_model_name:
        model_parent_dir = str(cli_model_parent_dir or inference_cfg.get("model_parent_dir", "model"))
        model_name = str(cli_model_name or inference_cfg.get("model_name", "multi-domain-rm-llama-3-8b-it"))
        return f"./{model_parent_dir}/{model_name}"

    model_parent_dir = str(inference_cfg.get("model_parent_dir", "model"))
    model_name = str(inference_cfg.get("model_name", "multi-domain-rm-llama-3-8b-it"))
    return f"./{model_parent_dir}/{model_name}"

def main() -> None:
    parser = ArgumentParser(description="Evaluate packaged reward model.")
    parser.add_argument("--config_path", type=str, default="config.yaml", help="Path to YAML config file.")
    parser.add_argument("--model_path", type=str, default=None, help="Optional override for packaged model path.")
    parser.add_argument("--model_parent_dir", type=str, default="model", help="Optional packaged model parent directory.")
    parser.add_argument("--model_name", type=str, default=None, help="Optional packaged model directory name.")
    parser.add_argument("--model_family", type=str, default=None, help="Model family (llama3, gemma2, qwen3, auto).")
    args = parser.parse_args()

    config = load_yaml_config(args.config_path)
    path = _resolve_inference_model_path(config, args.model_path, args.model_parent_dir, args.model_name)
    use_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if use_cuda else torch.float32

    print(f"Loading model: {path}")

    # `device_map="auto"` requires `_no_split_modules` in custom model classes.
    # For this project model, load on a single GPU when CUDA is available.
    model = RewardModelWithGating.from_pretrained(
        path,
        device_map={"": 0} if use_cuda else None,
        dtype=dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
    model.eval()

    device = next(model.parameters()).device

    # Example prompt/response to test empathy and multicultural sensitivity.
    prompt = "I just moved to Japan for work and I feel overwhelmed and lonely. Today I accidentally offended my manager because I did not know a local custom."
    response = "I am sorry you are going through this. Feeling overwhelmed after moving to a different culture is completely normal. Etiquette mistakes happen often, especially early on. If you want, we can walk through what happened with your manager and draft a respectful way to address it."

    messages = [{"role": "user", "content": prompt},
                {"role": "assistant", "content": response}]
    encoding = tokenizer.apply_chat_template(messages, return_tensors="pt", padding=True)
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding.get("attention_mask")
    attention_mask = attention_mask.to(device) if attention_mask is not None else None

    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        # Raw rewards for each of the 23 objectives.
        multi_obj_rewards = output.rewards.cpu().float()
        # Gating output (objective importance conditioned on the prompt).
        gating_output = output.gating_output.cpu().float()
        # Final preference score.
        preference_score = output.score.cpu().float()

    obj_transform = model.reward_transform_matrix.data.cpu().float()
    multi_obj_coeffs = gating_output @ obj_transform.T

    # Ensure the decomposition is numerically consistent (allowing small mixed-precision error).
    reconstructed_score = torch.sum(multi_obj_rewards * multi_obj_coeffs, dim=1)
    if not torch.allclose(reconstructed_score, preference_score, atol=1e-2, rtol=1e-3):
        max_abs_diff = torch.max(torch.abs(reconstructed_score - preference_score)).item()
        print(f"Warning: score decomposition mismatch (max abs diff = {max_abs_diff:.6f}).")

    K = 3
    top_obj_dims = torch.argsort(torch.abs(multi_obj_coeffs), dim=1, descending=True)[:, :K]
    top_obj_coeffs = torch.gather(multi_obj_coeffs, dim=1, index=top_obj_dims)

    # The order must match Stage 1 training exactly.
    attributes = [
        'co_discourse_structure', 'co_logical_consistency', 'co_mutual_grounding',
        'co_overall_coherence_score', 'co_temporal_causal_coherence', 'co_topic_coherence',
        'cs_causality', 'cs_coherence', 'cs_consistency', 'cs_desire', 'cs_empathy', 'cs_reaction',
        'em_emotional_awareness', 'em_emotional_validation', 'em_helpful_response',
        'em_overall_empathy_score', 'em_perspective_taking', 'em_supportive_engagement',
        'mu_coherence', 'mu_cultural_specificity', 'mu_cultural_value', 'mu_empathy', 'mu_naturalness'
    ]

    print("\n--- MULTI-DOMAIN EVALUATION ---")
    print(f"Global Preference Score: {preference_score.item():.5f}")
    print(f"\nTop {K} attributes driving this decision:")

    example_index = 0
    for i in range(K):
        attribute_idx = int(top_obj_dims[example_index, i].item())
        attribute = attributes[attribute_idx]
        coeff = top_obj_coeffs[example_index, i].item()
        print(f" - {attribute}: {round(coeff, 5)}")


if __name__ == "__main__":
    main()