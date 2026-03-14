import os
import sys
from typing import Any, Dict

import yaml


def cli_has_flag(flag: str, argv=None) -> bool:
    args = argv if argv is not None else sys.argv[1:]
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in args)


def apply_selected_model_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(config, dict):
        return config

    model_cfg = config.get("model")
    if not isinstance(model_cfg, dict):
        return config

    selected = model_cfg.get("selected")
    registry = model_cfg.get("registry", {})
    if not isinstance(registry, dict) or not selected or selected not in registry:
        return config

    selected_model_cfg = registry.get(selected, {})
    if not isinstance(selected_model_cfg, dict):
        return config

    # Project-level packaged model naming by profile.
    profile_name_map = {
        "llama3": "multi-domain-rm-llama-3-8b-it",
        "gemma2": "multi-domain-rm-gemma-2-9b-it",
        "qwen3": "multi-domain-rm-qwen-3-8b-it",
    }
    target_model_name = selected_model_cfg.get("packaged_model_name") or profile_name_map.get(selected)
    if not target_model_name:
        return config

    stage3_cfg = config.get("stage_3_package")
    if not isinstance(stage3_cfg, dict):
        stage3_cfg = {}
        config["stage_3_package"] = stage3_cfg
    stage3_cfg["output_model_name"] = str(target_model_name)

    inference_cfg = config.get("inference")
    if not isinstance(inference_cfg, dict):
        inference_cfg = {}
        config["inference"] = inference_cfg
    inference_cfg["model_name"] = str(target_model_name)

    return config


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    if not config_path:
        return {}
    if not os.path.isfile(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        return {}
    data = apply_selected_model_defaults(data)
    return data


def apply_section_overrides(args, section_cfg: Dict[str, Any], argv=None, skip_keys=None):
    if not section_cfg:
        return args
    skip = set(skip_keys or [])
    for key, value in section_cfg.items():
        if key in skip:
            continue
        if not hasattr(args, key):
            continue
        if value is None:
            continue
        flag = f"--{key}"
        if not cli_has_flag(flag, argv=argv):
            setattr(args, key, value)
    return args


def resolve_model_from_config(args, config: Dict[str, Any], needs_family: bool = False, argv=None):
    model_cfg = config.get("model") if isinstance(config, dict) else {}
    if not isinstance(model_cfg, dict):
        return args

    registry = model_cfg.get("registry", {})
    if not isinstance(registry, dict):
        registry = {}

    selected_key = None
    if hasattr(args, "model_key") and getattr(args, "model_key", None):
        selected_key = getattr(args, "model_key")
    elif not cli_has_flag("--model_key", argv=argv):
        selected_key = model_cfg.get("selected")

    if not selected_key or selected_key not in registry:
        return args

    selected_model = registry.get(selected_key, {})
    if not isinstance(selected_model, dict):
        return args

    if hasattr(args, "model_key"):
        args.model_key = selected_key

    if hasattr(args, "model_path") and not cli_has_flag("--model_path", argv=argv):
        model_path = selected_model.get("model_path")
        if model_path:
            args.model_path = model_path

    if needs_family and hasattr(args, "model_family") and not cli_has_flag("--model_family", argv=argv):
        model_family = selected_model.get("model_family")
        if model_family:
            args.model_family = model_family

    return args
