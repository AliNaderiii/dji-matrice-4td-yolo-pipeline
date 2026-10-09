"""Configuration loading and validation for the PPE detection protocol."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_SPLITS = {"train", "validation", "test"}


def _require_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration requires a mapping named '{key}'")
    return value


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load a detection configuration and reject ambiguous class/split schemas."""
    path = Path(config_path) if config_path else PROJECT_ROOT / "config.yaml"
    with path.open("r", encoding="utf-8-sig") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")

    dataset = _require_mapping(config, "dataset")
    model = _require_mapping(config, "model")
    training = _require_mapping(config, "training")
    runtime = _require_mapping(config, "runtime")

    if not isinstance(dataset.get("name"), str) or not dataset["name"].strip():
        raise ValueError("dataset.name must be a non-empty string")
    class_names = dataset.get("class_names")
    if not isinstance(class_names, list) or len(class_names) < 2:
        raise ValueError("dataset.class_names must be a list containing at least two classes")
    if any(not isinstance(name, str) or not name.strip() for name in class_names):
        raise ValueError("dataset.class_names entries must be non-empty strings")
    if len({name.lower() for name in class_names}) != len(class_names):
        raise ValueError("dataset.class_names must be unique case-insensitively")

    splits = dataset.get("splits")
    if not isinstance(splits, dict) or set(splits) != REQUIRED_SPLITS:
        raise ValueError("dataset.splits must define exactly train, validation, and test")
    if any(not isinstance(value, str) or not value.strip() for value in splits.values()):
        raise ValueError("dataset.splits values must be non-empty folder names")

    if not isinstance(dataset.get("root"), str) or not dataset["root"].strip():
        raise ValueError("dataset.root must be a non-empty path string")
    if not isinstance(dataset.get("image_size"), int) or dataset["image_size"] < 32:
        raise ValueError("dataset.image_size must be an integer of at least 32")
    if not isinstance(model.get("weights"), str) or not model["weights"].strip():
        raise ValueError("model.weights must identify an Ultralytics model or local weights file")
    if not isinstance(model.get("pretrained"), bool):
        raise ValueError("model.pretrained must be a boolean")
    if not isinstance(training.get("run_name"), str) or not training["run_name"].strip():
        raise ValueError("training.run_name must be a non-empty string")
    if not isinstance(training.get("epochs"), int) or training["epochs"] < 1:
        raise ValueError("training.epochs must be a positive integer")
    if not isinstance(training.get("batch_size"), int) or training["batch_size"] < 1:
        raise ValueError("training.batch_size must be a positive integer")
    if not isinstance(training.get("early_stopping_patience"), int) or training["early_stopping_patience"] < 0:
        raise ValueError("training.early_stopping_patience must be a non-negative integer")
    if not isinstance(runtime.get("seed"), int):
        raise ValueError("runtime.seed must be an integer")
    if not isinstance(runtime.get("deterministic"), bool):
        raise ValueError("runtime.deterministic must be a boolean")
    if not isinstance(runtime.get("num_workers"), int) or runtime["num_workers"] < 0:
        raise ValueError("runtime.num_workers must be a non-negative integer")
    if runtime.get("device") not in {"auto", "cpu", "cuda:0"}:
        raise ValueError("runtime.device must be one of auto, cpu, or cuda:0")
    return config


def config_sha256(config: dict[str, Any]) -> str:
    """Return a stable digest for the fully resolved YAML configuration."""
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
