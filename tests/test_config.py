from pathlib import Path

import pytest

from src.config import load_config


def valid_config() -> str:
    return """dataset:
  name: ppe
  root: data
  class_names: [person, hard_hat, no_hard_hat]
  splits: {train: train, validation: val, test: test}
  image_size: 640
model: {weights: yolov8n.pt, pretrained: true}
training: {run_name: baseline, batch_size: 4, epochs: 1, early_stopping_patience: 1}
runtime: {seed: 42, deterministic: true, num_workers: 0, device: cpu}
"""


def test_valid_config_loads(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(valid_config(), encoding="utf-8")
    assert load_config(path)["dataset"]["class_names"] == ["person", "hard_hat", "no_hard_hat"]


def test_duplicate_class_names_fail(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(valid_config().replace("no_hard_hat", "PERSON"), encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        load_config(path)


def test_missing_test_split_fails(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(valid_config().replace(", test: test", ""), encoding="utf-8")
    with pytest.raises(ValueError, match="splits"):
        load_config(path)
