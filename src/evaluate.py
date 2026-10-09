"""Evaluate a recorded PPE checkpoint once on the held-out YOLO test split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT, config_sha256, load_config
from .data import PROTOCOL, file_sha256, validate_dataset, write_ultralytics_yaml


def _numeric_metrics(results: Any) -> dict[str, float]:
    """Extract only metrics actually supplied by Ultralytics."""
    values = getattr(results, "results_dict", None)
    if not isinstance(values, dict):
        raise ValueError("Ultralytics did not expose a dataset-level metric dictionary")
    metrics: dict[str, float] = {}
    for key, value in values.items():
        try:
            metrics[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    if not metrics:
        raise ValueError("Ultralytics metric dictionary contained no numeric values")
    return metrics


def _recorded_manifest() -> dict[str, Any]:
    path = PROJECT_ROOT / "checkpoints" / "split_manifest.json"
    if not path.is_file():
        raise FileNotFoundError("No recorded split manifest. Train with src.train before evaluating.")
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError("Recorded split manifest is not a JSON object")
    return parsed


def _resolve_checkpoint(checkpoint_path: str | Path, metadata: dict[str, Any]) -> Path:
    requested = Path(checkpoint_path) if checkpoint_path else Path(metadata["best_checkpoint"])
    selected = requested if requested.is_absolute() else PROJECT_ROOT / requested
    if not selected.is_file():
        raise FileNotFoundError(f"Selected checkpoint is unavailable: {selected}")
    actual_digest = file_sha256(selected)
    if actual_digest != metadata.get("best_checkpoint_sha256"):
        raise ValueError("Selected checkpoint digest does not match the recorded selected checkpoint")
    return selected


def evaluate(checkpoint_path: str | Path = "", config_path: str | Path = "config.yaml") -> dict[str, Any]:
    """Evaluate the fixed selected checkpoint on test after integrity checks pass."""
    config = load_config(config_path)
    metadata_path = PROJECT_ROOT / "checkpoints" / "run_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError("No recorded training metadata. Train with src.train before evaluating.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format_version") != 2 or metadata.get("protocol") != PROTOCOL:
        raise ValueError("Checkpoint metadata does not satisfy the current held-out test protocol")
    if metadata.get("resolved_config_sha256") != config_sha256(config):
        raise ValueError("Current config differs from the config that produced the selected checkpoint")
    if metadata.get("dataset", {}).get("class_names") != config["dataset"]["class_names"]:
        raise ValueError("Checkpoint metadata and config disagree about the class schema")

    _, current_manifest = validate_dataset(config)
    recorded_manifest = _recorded_manifest()
    recorded_fingerprint = recorded_manifest.get("dataset_fingerprint_sha256")
    if not recorded_fingerprint or recorded_fingerprint != metadata.get("dataset_fingerprint_sha256"):
        raise ValueError("Recorded metadata and split manifest disagree about dataset provenance")
    if current_manifest["dataset_fingerprint_sha256"] != recorded_fingerprint:
        raise ValueError("Current local dataset differs from the dataset used for training")
    if metadata.get("split_counts") != current_manifest["split_counts"]:
        raise ValueError("Current dataset split counts differ from the recorded training manifest")

    selected = _resolve_checkpoint(checkpoint_path, metadata)
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - optional runtime package
        raise RuntimeError("Install Ultralytics before evaluation: pip install -r requirements.txt") from exc

    dataset_yaml = write_ultralytics_yaml(config, PROJECT_ROOT / ".runtime" / "dataset.yaml")
    model = YOLO(str(selected))
    configured_device = config["runtime"]["device"]
    results = model.val(
        data=str(dataset_yaml),
        split="test",
        imgsz=int(config["dataset"]["image_size"]),
        batch=int(config["training"]["batch_size"]),
        workers=int(config["runtime"]["num_workers"]),
        device=None if configured_device == "auto" else configured_device,
    )
    output = {
        "format_version": 2,
        "split": "held_out_test",
        "protocol": PROTOCOL,
        "checkpoint": str(selected),
        "checkpoint_sha256": metadata["best_checkpoint_sha256"],
        "resolved_config_sha256": metadata["resolved_config_sha256"],
        "dataset_fingerprint_sha256": recorded_fingerprint,
        "split_counts": metadata["split_counts"],
        "class_names": config["dataset"]["class_names"],
        "metrics": _numeric_metrics(results),
    }
    reports = PROJECT_ROOT / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "test_metrics.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="", help="Must hash-match the recorded selected checkpoint.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.checkpoint, args.config), indent=2))
