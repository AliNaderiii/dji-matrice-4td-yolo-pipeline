#!/usr/bin/env python3
"""
Validate REAL Hard Hat Workers dataset (Roboflow layout)
Usage:
    python scripts/validate_real.py --root datasets/dji-hardhat-real --num-classes 3
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset.validator import DatasetValidator
from loguru import logger
from collections import Counter
import glob

def analyze_labels(root: Path, num_classes: int):
    """Quick analysis of label distribution"""
    root = Path(root)
    for split in ['train', 'valid', 'val', 'test']:
        # Try both layouts
        possible_label_dirs = [
            root / split / "labels",
            root / "labels" / split,
            root / "labels" / ("val" if split=="valid" else split),
        ]
        lbl_dir = None
        for p in possible_label_dirs:
            if p.exists():
                lbl_dir = p
                break
        if not lbl_dir:
            continue
        
        counter = Counter()
        total_boxes = 0
        files = list(lbl_dir.glob("*.txt"))
        for f in files:
            try:
                with open(f) as fp:
                    for line in fp:
                        line=line.strip()
                        if not line:
                            continue
                        cls = int(float(line.split()[0]))
                        counter[cls] += 1
                        total_boxes += 1
            except:
                pass
        
        logger.info(f"{split}: {len(files)} label files, {total_boxes} boxes, dist={dict(counter)}")
        
        # Check if classes are 0,1,2 or missing
        if counter:
            max_cls = max(counter.keys())
            if max_cls >= num_classes:
                logger.error(f"{split}: Found class {max_cls} but num_classes={num_classes} -> out of range!")
            missing = set(range(num_classes)) - set(counter.keys())
            if missing:
                logger.warning(f"{split}: Missing classes {missing} in this split (might be OK if rare)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="datasets/dji-hardhat-real", help="Dataset root containing train/ valid/")
    parser.add_argument("--num-classes", type=int, default=3, help="3 for real dataset (Person, Hard-Hat, No-Hard-Hat)")
    args = parser.parse_args()

    logger.info(f"Validating real dataset at {args.root}")
    
    # Quick label analysis first
    analyze_labels(Path(args.root), args.num_classes)
    
    # Full validation
    validator = DatasetValidator(dataset_root=args.root)
    ok = validator.run_full_validation(num_classes=args.num_classes)
    
    if ok:
        logger.success("Validation PASSED")
    else:
        logger.error("Validation FAILED - check logs above")
        # Don't exit with error for now, just show analysis

if __name__ == "__main__":
    main()
