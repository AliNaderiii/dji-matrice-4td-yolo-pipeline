"""Strict local YOLO-format dataset discovery and split provenance."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
PROTOCOL = "dataset-defined train/validation/test; test split held out until src.evaluate"


@dataclass(frozen=True)
class DetectionRecord:
    """One image and its optional YOLO label file.

    A missing label file is explicit negative-image supervision, which YOLO
    supports. Orphan label files and malformed labels are always rejected.
    """

    image_path: Path
    label_path: Path | None
    sample_id: str


def _image_files(folder: Path) -> list[Path]:
    return sorted(
        path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _validate_image(path: Path) -> None:
    """Reject unreadable image files before an expensive training job starts."""
    try:
        with Image.open(path) as image:
            image.verify()
    except (OSError, ValueError) as exc:
        raise ValueError(f"Unreadable image file: {path}") from exc


def _parse_label(path: Path, num_classes: int) -> Counter[int]:
    classes: Counter[int] = Counter()
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{path}:{line_number}: expected 5 YOLO fields, found {len(parts)}")
        try:
            class_id = int(parts[0])
            x_center, y_center, width, height = (float(value) for value in parts[1:])
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: YOLO fields must be numeric") from exc
        if not 0 <= class_id < num_classes:
            raise ValueError(
                f"{path}:{line_number}: class {class_id} is outside [0, {num_classes - 1}]"
            )
        if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0 and 0.0 < width <= 1.0 and 0.0 < height <= 1.0):
            raise ValueError(f"{path}:{line_number}: bounding box is outside normalized YOLO range")
        classes[class_id] += 1
    return classes


def discover_records(data_root: str | Path, split: str, num_classes: int) -> tuple[list[DetectionRecord], Counter[int]]:
    """Discover and validate one explicit split under ``images/<split>``."""
    root = Path(data_root)
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(
            f"Expected YOLO directories '{image_dir}' and '{label_dir}'. "
            "Data must be supplied locally; see docs/PROTOCOL.md."
        )
    images = _image_files(image_dir)
    if not images:
        raise FileNotFoundError(f"No supported images found in {image_dir}")
    labels_by_stem: dict[str, Path] = {}
    for label in sorted(label_dir.glob("*.txt")):
        key = label.stem.lower()
        if key in labels_by_stem:
            raise ValueError(f"Duplicate label stem '{label.stem}' in {label_dir}")
        labels_by_stem[key] = label

    records: list[DetectionRecord] = []
    class_counts: Counter[int] = Counter()
    image_stems: set[str] = set()
    for image in images:
        key = image.stem.lower()
        if key in image_stems:
            raise ValueError(f"Duplicate image stem '{image.stem}' in {image_dir}")
        image_stems.add(key)
        _validate_image(image)
        label = labels_by_stem.pop(key, None)
        if label is not None:
            class_counts.update(_parse_label(label, num_classes))
        records.append(DetectionRecord(image, label, image.stem))
    if labels_by_stem:
        examples = ", ".join(sorted(path.name for path in labels_by_stem.values())[:5])
        raise FileNotFoundError(f"{len(labels_by_stem)} orphan YOLO label file(s) in {label_dir}: {examples}")
    return records, class_counts


def _is_negative(record: DetectionRecord) -> bool:
    """Return whether a record has no annotated objects (missing or empty label)."""
    return record.label_path is None or not record.label_path.read_text(encoding="utf-8").strip()


def _inventory(data_root: Path, records: dict[str, list[DetectionRecord]]) -> dict[str, list[dict[str, str | None]]]:
    """Hash every declared input so a later evaluation can reject data drift."""
    inventory: dict[str, list[dict[str, str | None]]] = {}
    for split, values in records.items():
        inventory[split] = [
            {
                "sample_id": record.sample_id,
                "image": str(record.image_path.relative_to(data_root)),
                "image_sha256": file_sha256(record.image_path),
                "label": str(record.label_path.relative_to(data_root)) if record.label_path else None,
                "label_sha256": file_sha256(record.label_path) if record.label_path else None,
            }
            for record in values
        ]
    return inventory


def validate_dataset(config: dict[str, Any]) -> tuple[dict[str, list[DetectionRecord]], dict[str, Any]]:
    """Validate all three splits and return a content-addressed run manifest."""
    dataset = config["dataset"]
    class_names = dataset["class_names"]
    num_classes = len(class_names)
    data_root = Path(dataset["root"])
    splits = dataset["splits"]
    records: dict[str, list[DetectionRecord]] = {}
    class_counts: dict[str, Counter[int]] = {}
    for protocol_split, folder_name in splits.items():
        records[protocol_split], class_counts[protocol_split] = discover_records(
            data_root, folder_name, num_classes
        )

    identifiers = {split: {record.sample_id.lower() for record in values} for split, values in records.items()}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = identifiers[left] & identifiers[right]
        if overlap:
            example = sorted(overlap)[0]
            raise ValueError(f"Dataset split overlap between {left} and {right}: {example}")

    inventory = _inventory(data_root, records)
    fingerprint_input = {
        "class_names": class_names,
        "split_folders": splits,
        "inventory": inventory,
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_input, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    manifest = {
        "format_version": 2,
        "dataset": dataset["name"],
        "protocol": PROTOCOL,
        "class_names": class_names,
        "data_root": str(data_root.resolve()),
        "split_folders": splits,
        "split_counts": {split: len(values) for split, values in records.items()},
        "class_box_counts": {
            split: {class_names[index]: int(counts.get(index, 0)) for index in range(num_classes)}
            for split, counts in class_counts.items()
        },
        "negative_image_counts": {
            split: sum(_is_negative(record) for record in values) for split, values in records.items()
        },
        "train_ids": sorted(record.sample_id for record in records["train"]),
        "validation_ids": sorted(record.sample_id for record in records["validation"]),
        "test_ids": sorted(record.sample_id for record in records["test"]),
        "inventory": inventory,
        "dataset_fingerprint_sha256": fingerprint,
    }
    return records, manifest


def write_manifest(manifest: dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def write_ultralytics_yaml(config: dict[str, Any], output_path: str | Path) -> Path:
    """Write a temporary local Ultralytics YAML from validated project config."""
    dataset = config["dataset"]
    root = Path(dataset["root"]).resolve()
    payload = {
        "path": str(root),
        "train": f"images/{dataset['splits']['train']}",
        "val": f"images/{dataset['splits']['validation']}",
        "test": f"images/{dataset['splits']['test']}",
        "names": {index: name for index, name in enumerate(dataset["class_names"])},
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import yaml

    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
