from pathlib import Path

import pytest
from PIL import Image

from src.data import validate_dataset, write_ultralytics_yaml


def write_split(root: Path, split: str, stem: str, label: str | None = "0 0.5 0.5 0.2 0.2") -> None:
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color="white").save(image_dir / f"{stem}.jpg")
    if label is not None:
        (label_dir / f"{stem}.txt").write_text(label + "\n", encoding="utf-8")


def config(root: Path) -> dict:
    return {
        "dataset": {
            "name": "ppe",
            "root": str(root),
            "class_names": ["person", "hard_hat", "no_hard_hat"],
            "splits": {"train": "train", "validation": "val", "test": "test"},
            "image_size": 640,
        }
    }


def test_manifest_is_strict_and_preserves_negative_images(tmp_path: Path) -> None:
    write_split(tmp_path, "train", "train_person")
    write_split(tmp_path, "train", "train_negative", label=None)
    write_split(tmp_path, "val", "val_hat", "1 0.5 0.5 0.2 0.2")
    write_split(tmp_path, "test", "test_no_hat", "2 0.5 0.5 0.2 0.2")

    records, manifest = validate_dataset(config(tmp_path))
    assert manifest["split_counts"] == {"train": 2, "validation": 1, "test": 1}
    assert manifest["negative_image_counts"]["train"] == 1
    assert manifest["class_box_counts"]["test"]["no_hard_hat"] == 1
    assert len(records["train"]) == 2
    assert len(manifest["dataset_fingerprint_sha256"]) == 64


def test_dataset_fingerprint_detects_local_data_change(tmp_path: Path) -> None:
    write_split(tmp_path, "train", "train")
    write_split(tmp_path, "val", "val")
    write_split(tmp_path, "test", "test")
    _, before = validate_dataset(config(tmp_path))
    (tmp_path / "labels" / "test" / "test.txt").write_text("1 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    _, after = validate_dataset(config(tmp_path))
    assert before["dataset_fingerprint_sha256"] != after["dataset_fingerprint_sha256"]


def test_orphan_label_fails(tmp_path: Path) -> None:
    write_split(tmp_path, "train", "train")
    write_split(tmp_path, "val", "val")
    write_split(tmp_path, "test", "test")
    (tmp_path / "labels" / "train" / "orphan.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="orphan"):
        validate_dataset(config(tmp_path))


def test_unreadable_image_fails(tmp_path: Path) -> None:
    write_split(tmp_path, "train", "train")
    write_split(tmp_path, "val", "val")
    write_split(tmp_path, "test", "test")
    (tmp_path / "images" / "train" / "train.jpg").write_bytes(b"not an image")
    with pytest.raises(ValueError, match="Unreadable"):
        validate_dataset(config(tmp_path))


def test_out_of_range_yolo_class_fails(tmp_path: Path) -> None:
    write_split(tmp_path, "train", "train", "3 0.5 0.5 0.2 0.2")
    write_split(tmp_path, "val", "val")
    write_split(tmp_path, "test", "test")
    with pytest.raises(ValueError, match="outside"):
        validate_dataset(config(tmp_path))


def test_ultralytics_yaml_has_one_authoritative_class_map(tmp_path: Path) -> None:
    output = write_ultralytics_yaml(config(tmp_path), tmp_path / "runtime.yaml")
    text = output.read_text(encoding="utf-8")
    assert "person" in text
    assert "hard_hat" in text
    assert "no_hard_hat" in text
