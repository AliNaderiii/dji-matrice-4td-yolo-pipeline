"""Generate a dashboard only from recorded YOLO history and held-out metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import PROJECT_ROOT


def _series(history: list[dict[str, Any]], candidates: list[str]) -> list[float]:
    for key in candidates:
        if all(key in row and isinstance(row[key], (int, float)) for row in history):
            return [float(row[key]) for row in history]
    raise ValueError(f"Training history lacks a complete metric series. Tried {candidates}")


def build_dashboard(metadata_path: str | Path, metrics_path: str | Path, output_path: str | Path) -> None:
    """Build a plot from actual metadata only; no values are hard-coded."""
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    evaluation = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    history = metadata.get("history", [])
    if not history:
        raise ValueError("Recorded run metadata contains no parsed Ultralytics history")
    epochs = list(range(1, len(history) + 1))
    train_loss = _series(history, ["train/box_loss", "train/box_loss "])
    val_map50 = _series(history, ["metrics/mAP50(B)", "metrics/mAP50"])
    val_map5095 = _series(history, ["metrics/mAP50-95(B)", "metrics/mAP50-95"])
    heldout = evaluation["metrics"]

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    axes[0].plot(epochs, train_loss, linewidth=2, color="#2563eb")
    axes[0].set(title="Training box loss", xlabel="Epoch", ylabel="Loss")
    axes[0].grid(alpha=0.25)

    axes[1].plot(epochs, val_map50, label="development validation mAP50", linewidth=2)
    axes[1].plot(epochs, val_map5095, label="development validation mAP50-95", linewidth=2)
    axes[1].set(title="Development validation", xlabel="Epoch", ylabel="mAP")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    labels = ["precision", "recall", "mAP50", "mAP50-95"]
    metric_keys = ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]
    missing = [key for key in metric_keys if not isinstance(heldout.get(key), (int, float))]
    if missing:
        raise ValueError(f"Held-out metrics are missing required keys: {', '.join(missing)}")
    values = [float(heldout[key]) for key in metric_keys]
    axes[2].bar(labels, values, color=["#2563eb", "#059669", "#d97706", "#7c3aed"])
    axes[2].set(title="Held-out test metrics", ylabel="Score", ylim=(0, 1))
    axes[2].tick_params(axis="x", rotation=25)
    axes[2].grid(axis="y", alpha=0.25)

    figure.suptitle("Industrial PPE YOLO experiment — held-out metrics from the recorded checkpoint")
    figure.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", default=str(PROJECT_ROOT / "checkpoints" / "run_metadata.json"))
    parser.add_argument("--metrics", default=str(PROJECT_ROOT / "reports" / "test_metrics.json"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "reports" / "experiment_dashboard.png"))
    args = parser.parse_args()
    build_dashboard(args.metadata, args.metrics, args.output)
