from typing import Dict, List
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

class MultiDomainRMPipeline:
    def __init__(self, model_id, device_map="auto", torch_dtype=None, truncation=True, max_length=4096):
        if torch_dtype is None:
            torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        # `device_map="auto"` is not supported by this custom model class because
        # Transformers requires `_no_split_modules` for automatic sharding.
        if device_map == "auto":
            device_map = {"": 0} if torch.cuda.is_available() else None

        self.model = RewardModelWithGating.from_pretrained(
            model_id,
            device_map=device_map,
            dtype=torch_dtype,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
        )
        self.truncation = truncation
        self.device = next(self.model.parameters()).device
        self.max_length = max_length
        self.model.eval()

    def __call__(self, messages: List[Dict[str, str]]) -> Dict[str, float]:
        encoding = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            padding=True,
            truncation=self.truncation,
            max_length=self.max_length,
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding.get("attention_mask")
        attention_mask = attention_mask.to(self.device) if attention_mask is not None else None
        with torch.no_grad():
            output = self.model(input_ids=input_ids, attention_mask=attention_mask)
            score = output.score.float().item()
        return {"score": score}


def main() -> None:
    parser = ArgumentParser(description="Run quick prediction comparison using packaged reward model.")
    parser.add_argument("--config_path", type=str, default="config.yaml", help="Path to YAML config file.")
    parser.add_argument("--model_path", type=str, default=None, help="Optional override for packaged model path.")
    parser.add_argument("--model_parent_dir", type=str, default="model", help="Optional packaged model parent directory.")
    parser.add_argument("--model_name", type=str, default=None, help="Optional packaged model directory name.")
    parser.add_argument("--model_family", type=str, default=None, help="Model family (llama3, gemma2, qwen3, auto).")
    args = parser.parse_args()

    config = load_yaml_config(args.config_path)
    model_path = _resolve_inference_model_path(config, args.model_path, args.model_parent_dir, args.model_name)
    print(f"Loading model: {model_path}")

    # Initialize the local model.
    rm = MultiDomainRMPipeline(model_path)

    prompt = "I just moved to Japan for work and I feel overwhelmed and lonely."

    # Response 1: high empathy and naturalness.
    response1 = "It makes complete sense that you feel this way. Relocating is a major life transition. Give yourself time to adapt."
    score1 = rm([{"role": "user", "content": prompt}, {"role": "assistant", "content": response1}])
    print(f"Response 1 (Empathetic) - Score: {score1['score']:.5f}")

    # Response 2: colder and more mechanical style.
    response2 = "To solve loneliness, join expat groups and study the language for two hours every day."
    score2 = rm([{"role": "user", "content": prompt}, {"role": "assistant", "content": response2}])
    print(f"Response 2 (Robotic) - Score: {score2['score']:.5f}")


if __name__ == "__main__":
    main()