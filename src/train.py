"""Train an Ultralytics PPE detector without iterating the held-out test split."""
from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from .config import PROJECT_ROOT, config_sha256, load_config
from .data import PROTOCOL, file_sha256, validate_dataset, write_manifest, write_ultralytics_yaml


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def set_seed(seed: int) -> None:
    """Set Python/NumPy seed before the external trainer receives its own seed."""
    random.seed(seed)
    np.random.seed(seed)


def _history_from_results_csv(path: Path) -> list[dict[str, Any]]:
    """Parse Ultralytics' generated history without inventing missing values."""
    if not path.is_file():
        return []
    history: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            normalized: dict[str, Any] = {}
            for key, value in row.items():
                if key is None:
                    continue
                try:
                    normalized[key.strip()] = float(value)
                except (TypeError, ValueError):
                    normalized[key.strip()] = value
            history.append(normalized)
    return history


def train(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    """Validate local data, train on train/validation only, and persist provenance."""
    config = load_config(config_path)
    set_seed(int(config["runtime"]["seed"]))
    _, manifest = validate_dataset(config)
    checkpoints = PROJECT_ROOT / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    write_manifest(manifest, checkpoints / "split_manifest.json")
    resolved_config_path = checkpoints / "resolved_config.json"
    resolved_config_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    resolved_config_sha256 = config_sha256(config)

    try:
        import torch
        import ultralytics
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - depends on optional runtime package
        raise RuntimeError("Install Ultralytics before training: pip install -r requirements.txt") from exc

    run_name = config["training"]["run_name"]
    runs_root = PROJECT_ROOT / "runs"
    run_dir = runs_root / run_name
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite a prior run directory: {run_dir}")

    dataset_yaml = write_ultralytics_yaml(config, PROJECT_ROOT / ".runtime" / "dataset.yaml")
    model = YOLO(config["model"]["weights"])
    configured_device = config["runtime"]["device"]
    model.train(
        data=str(dataset_yaml),
        epochs=int(config["training"]["epochs"]),
        imgsz=int(config["dataset"]["image_size"]),
        batch=int(config["training"]["batch_size"]),
        patience=int(config["training"]["early_stopping_patience"]),
        workers=int(config["runtime"]["num_workers"]),
        seed=int(config["runtime"]["seed"]),
        deterministic=bool(config["runtime"]["deterministic"]),
        pretrained=bool(config["model"]["pretrained"]),
        device=None if configured_device == "auto" else configured_device,
        project=str(runs_root),
        name=run_name,
        exist_ok=False,
        plots=True,
        save=True,
        val=True,
    )

    best_checkpoint = run_dir / "weights" / "best.pt"
    if not best_checkpoint.is_file():
        raise FileNotFoundError(f"Ultralytics training completed without best.pt at {best_checkpoint}")
    checkpoint_sha256 = file_sha256(best_checkpoint)
    (checkpoints / "checkpoint.sha256").write_text(
        f"{checkpoint_sha256}  {best_checkpoint.name}\n", encoding="utf-8"
    )
    metadata = {
        "format_version": 2,
        "protocol": PROTOCOL,
        "code_commit": _git_commit(),
        "config_path": str(config_path),
        "resolved_config_path": str(resolved_config_path.relative_to(PROJECT_ROOT)),
        "resolved_config_sha256": resolved_config_sha256,
        "dataset_fingerprint_sha256": manifest["dataset_fingerprint_sha256"],
        "model": config["model"],
        "dataset": {
            "name": config["dataset"]["name"],
            "class_names": config["dataset"]["class_names"],
            "image_size": config["dataset"]["image_size"],
        },
        "split_counts": manifest["split_counts"],
        "run_dir": str(run_dir.relative_to(PROJECT_ROOT)),
        "best_checkpoint": str(best_checkpoint.relative_to(PROJECT_ROOT)),
        "best_checkpoint_sha256": checkpoint_sha256,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "history": _history_from_results_csv(run_dir / "results.csv"),
    }
    (checkpoints / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    result = train(args.config)
    print(json.dumps({"best_checkpoint": result["best_checkpoint"], "split_counts": result["split_counts"]}, indent=2))
