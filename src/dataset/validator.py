"""
Dataset Validator - Ensures YOLO format correctness for DJI deployment
Author: Ali Naderi | Edge AI Engineer
Supports both layouts:
  - Roboflow: train/images, train/labels, valid/images, valid/labels
  - Custom:   images/train, labels/train, images/val, labels/val
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
from loguru import logger
import cv2
from collections import Counter


class DatasetValidator:
    """
    Validates YOLO dataset before training.
    Critical for DJI: wrong format causes quantization failure.
    Auto-detects Roboflow vs custom layout.
    """

    def __init__(self, dataset_root: str = "./datasets/dji-hardhat"):
        self.root = Path(dataset_root).resolve()
        self.layout = self._detect_layout()
        logger.info(f"Dataset root: {self.root}")
        logger.info(f"Detected layout: {self.layout}")

    def _detect_layout(self) -> str:
        """Detect whether dataset uses Roboflow or custom layout"""
        # Roboflow layout: train/images, valid/images
        if (self.root / "train" / "images").exists() or (self.root / "train").exists():
            # Check if train/images exists OR train contains images directly?
            if (self.root / "train" / "images").exists():
                return "roboflow"  # train/images, valid/images, test/images
            # Fallback check for images inside train folder directly (some exports)
            if (self.root / "train").exists() and any((self.root / "train").glob("*.jpg")):
                return "roboflow_flat"
        # Custom layout: images/train, labels/train
        if (self.root / "images" / "train").exists() or (self.root / "images").exists():
            return "custom"
        # If nothing matches, try to infer from data.yaml
        yaml_candidates = list(self.root.glob("*.yaml")) + list(self.root.glob("*.yml"))
        for y in yaml_candidates:
            try:
                import yaml
                with open(y, 'r') as f:
                    data = yaml.safe_load(f)
                if 'train' in data:
                    # If train contains 'images', it's custom, if contains 'train/images' it's roboflow-like
                    return "yaml_defined"
            except:
                pass
        return "unknown"

    def _resolve_split_paths(self, split: str) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Resolve image and label dirs for a given split.
        Supports: train, val, valid, test
        """
        # Normalize split name: valid <-> val
        split_variants = {
            'train': ['train'],
            'val': ['val', 'valid'],
            'valid': ['valid', 'val'],
            'test': ['test']
        }
        candidates = split_variants.get(split, [split])

        for cand in candidates:
            # Try Roboflow layout first: <root>/<split>/images, <root>/<split>/labels
            img_robo = self.root / cand / "images"
            lbl_robo = self.root / cand / "labels"
            if img_robo.exists():
                # Label might be in same folder or labels folder
                if lbl_robo.exists():
                    return img_robo, lbl_robo
                # Some exports have labels in <split>/labels but images in <split>/images - already covered
                # Fallback: check if labels exist as txt alongside?
                return img_robo, lbl_robo

            # Try custom layout: <root>/images/<split>, <root>/labels/<split>
            img_custom = self.root / "images" / cand
            lbl_custom = self.root / "labels" / cand
            if img_custom.exists():
                return img_custom, lbl_custom

            # Flat layout: <root>/<cand> contains images directly
            flat_dir = self.root / cand
            if flat_dir.exists() and any(flat_dir.glob("*.jpg")):
                # Labels might be in same dir or in labels subfolder?
                lbl_flat = flat_dir / "labels" if (flat_dir / "labels").exists() else flat_dir
                return flat_dir, lbl_flat

        # Not found
        return None, None

    def _count_images(self, img_dir: Path) -> int:
        if not img_dir or not img_dir.exists():
            return 0
        return len(list(img_dir.glob("*.jpg"))) + len(list(img_dir.glob("*.jpeg"))) + \
               len(list(img_dir.glob("*.png"))) + len(list(img_dir.glob("*.bmp")))

    def _count_labels(self, lbl_dir: Path) -> int:
        if not lbl_dir or not lbl_dir.exists():
            return 0
        return len(list(lbl_dir.glob("*.txt")))

    def validate_structure(self) -> bool:
        """Check train/val/test folders exist"""
        success = True
        for split in ['train', 'val']:
            img_dir, lbl_dir = self._resolve_split_paths(split)
            
            if img_dir is None or not img_dir.exists():
                logger.error(f"Missing {split} images folder. Tried: {self.root / split / 'images'} and {self.root / 'images' / split}")
                success = False
                continue
            
            if lbl_dir is None or not lbl_dir.exists():
                logger.warning(f"Missing {split} labels folder: {lbl_dir} - will check if labels are alongside images")
                # Try alternative: labels might be in same dir as images for some datasets? No, but warn
                # For roboflow, labels should exist
                if split == 'train':
                    success = False
                    continue

            img_count = self._count_images(img_dir)
            lbl_count = self._count_labels(lbl_dir) if lbl_dir else 0

            logger.info(f"{split}: {img_count} images in {img_dir}, {lbl_count} labels in {lbl_dir}")

            if img_count == 0:
                logger.warning(f"No images in {split} ({img_dir})")
                if split == 'train':
                    success = False

        # Test is optional
        img_test, lbl_test = self._resolve_split_paths('test')
        if img_test and img_test.exists():
            logger.info(f"test: {self._count_images(img_test)} images in {img_test}")

        return success

    def validate_labels(self, split: str = "train", num_classes: int = 4) -> Dict:
        """
        Validate YOLO label format:
        - Each line: class_id x_center y_center width height (normalized 0-1)
        - class_id in [0, num_classes-1]
        """
        img_dir, lbl_dir = self._resolve_split_paths(split)
        
        stats = {
            'total_files': 0,
            'valid_files': 0,
            'invalid_files': [],
            'class_distribution': Counter(),
            'bbox_issues': [],
            'image_dir': str(img_dir) if img_dir else None,
            'label_dir': str(lbl_dir) if lbl_dir else None,
        }

        if not lbl_dir or not lbl_dir.exists():
            # Try to find labels alongside images or in alternative location
            logger.warning(f"Label dir not found for {split}, trying alternatives")
            # Check if txt files exist in image dir (unlikely but possible)
            if img_dir and img_dir.exists():
                txt_in_img = list(img_dir.glob("*.txt"))
                if txt_in_img:
                    lbl_dir = img_dir
                else:
                    logger.error(f"No label dir for {split}")
                    return stats

        for label_file in lbl_dir.glob("*.txt"):
            stats['total_files'] += 1
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()

                if not lines:
                    continue

                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) != 5:
                        stats['invalid_files'].append(f"{label_file.name}:{line_num} - wrong format ({len(parts)} cols)")
                        continue

                    try:
                        cls_id = int(float(parts[0]))  # Handle float class ids sometimes
                        x_c, y_c, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    except ValueError:
                        stats['invalid_files'].append(f"{label_file.name}:{line_num} - not numeric")
                        continue

                    # Checks
                    if not (0 <= cls_id < num_classes):
                        stats['invalid_files'].append(f"{label_file.name}:{line_num} - class {cls_id} out of range [0,{num_classes-1}]")
                        continue

                    if not (0 <= x_c <= 1 and 0 <= y_c <= 1 and 0 < w <= 1 and 0 < h <= 1):
                        stats['bbox_issues'].append(f"{label_file.name}:{line_num} - bbox out of [0,1]: {x_c},{y_c},{w},{h}")
                        continue

                    # Additional sanity: bbox should not be extremely small
                    if w < 0.001 or h < 0.001:
                        stats['bbox_issues'].append(f"{label_file.name}:{line_num} - tiny bbox {w}x{h}")

                    stats['class_distribution'][cls_id] += 1
                    stats['valid_files'] += 1

            except Exception as e:
                stats['invalid_files'].append(f"{label_file.name} - {e}")

        logger.info(f"Validation {split}: {stats['total_files']} label files, {stats['valid_files']} valid boxes")
        logger.info(f"Class distribution {split}: {dict(stats['class_distribution'])}")

        if stats['invalid_files']:
            logger.warning(f"Found {len(stats['invalid_files'])} invalid entries in {split} (first 5): {stats['invalid_files'][:5]}")
        if stats['bbox_issues']:
            logger.warning(f"Found {len(stats['bbox_issues'])} bbox issues in {split} (first 5): {stats['bbox_issues'][:5]}")

        return stats

    def check_class_imbalance(self, train_stats: Dict, threshold: float = 0.1):
        """Warn if any class has <10% of total"""
        total = sum(train_stats['class_distribution'].values())
        if total == 0:
            logger.warning("No labels found!")
            return

        for cls_id, count in train_stats['class_distribution'].items():
            ratio = count / total
            if ratio < threshold:
                logger.warning(f"Class {cls_id} is under-represented: {count}/{total} ({ratio:.1%}) - consider augmentation")

        # Check missing classes
        if 'num_classes' in train_stats:
            expected = train_stats['num_classes']
            found = set(train_stats['class_distribution'].keys())
            missing = set(range(expected)) - found
            if missing:
                logger.warning(f"Missing classes in training set: {missing}")

    def run_full_validation(self, num_classes: int = 4) -> bool:
        """Run all checks"""
        logger.info("=== Dataset Validation ===")
        logger.info(f"Root: {self.root}, Expected classes: {num_classes}")

        if not self.validate_structure():
            logger.error("Structure validation failed - trying to continue with label checks anyway for debug")
            # Don't return False immediately, try to show more info

        train_stats = self.validate_labels('train', num_classes)
        train_stats['num_classes'] = num_classes

        # Try val/valid
        val_stats = self.validate_labels('val', num_classes)
        if val_stats['total_files'] == 0:
            val_stats = self.validate_labels('valid', num_classes)

        self.check_class_imbalance(train_stats)

        # Final report
        if train_stats['total_files'] == 0:
            logger.error("No training labels found! Check dataset path and structure.")
            logger.info(f"Tip: For Roboflow layout, use path containing train/images, valid/images")
            logger.info(f"     Example: datasets/dji-hardhat-real (which contains train/, valid/, test/)")
            return False

        if train_stats['valid_files'] == 0:
            logger.error("No valid boxes found!")
            return False

        logger.success(f"Dataset validation completed: {train_stats['valid_files']} boxes, {train_stats['total_files']} files")
        return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="./datasets/dji-hardhat", help="Dataset root")
    parser.add_argument("--num-classes", type=int, default=4, help="Number of classes")
    args = parser.parse_args()

    validator = DatasetValidator(dataset_root=args.root)
    ok = validator.run_full_validation(num_classes=args.num_classes)
    if not ok:
        exit(1)
