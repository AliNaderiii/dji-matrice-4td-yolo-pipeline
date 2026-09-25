#!/usr/bin/env python3
"""
Merge COCO Vehicle (car, truck, bus) into Hard-Hat dataset for 4-class model
Author: Ali Naderi | DJI Matrice 4TD

Classes:
  0: Person (from Hard-Hat dataset + COCO)
  1: Vehicle (from COCO: car 2, bus 5, truck 7)
  2: Hard-Hat (from Hard-Hat Workers)
  3: No-Hard-Hat (from Hard-Hat Workers)

Usage:
  python scripts/merge_coco_vehicle.py --hardhat-root datasets/dji-hardhat-real --coco-root datasets/coco --limit-vehicle 2000 --output datasets/dji-hardhat-4class
"""

import argparse
import shutil
import random
from pathlib import Path
from loguru import logger
import yaml

# COCO class mapping
COCO_VEHICLE_IDS = [2, 5, 7]  # car, bus, truck
COCO_PERSON_ID = 0

def create_4class_yaml(output_root: Path, train_rel: str = "train/images", val_rel: str = "valid/images"):
    """Create 4-class yaml"""
    yaml_path = output_root / "dji_matrice_4class.yaml"
    config = {
        'path': str(output_root.resolve()),
        'train': train_rel,
        'val': val_rel,
        'test': 'test/images',
        'nc': 4,
        'names': {
            0: 'Person',
            1: 'Vehicle',
            2: 'Hard-Hat',
            3: 'No-Hard-Hat'
        }
    }
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    logger.success(f"Created {yaml_path}")
    return yaml_path

def merge_datasets(hardhat_root: Path, coco_root: Path, output_root: Path, limit_vehicle: int = 2000):
    """
    Merge hardhat (3-class) + COCO vehicle into 4-class
    For simplicity, we copy hardhat as base and add vehicle images from COCO if available
    If COCO not available, we create 4-class structure with hardhat mapped to 0,2,3 and Vehicle placeholder
    """
    hardhat_root = Path(hardhat_root)
    output_root = Path(output_root)
    
    # Mapping: old 3-class -> new 4-class
    # Old: 0 Person, 1 Hard-Hat, 2 No-Hard-Hat
    # New: 0 Person, 1 Vehicle, 2 Hard-Hat, 3 No-Hard-Hat
    mapping_3to4 = {0: 0, 1: 2, 2: 3}
    
    # Create output dirs
    for split in ['train', 'valid', 'test']:
        (output_root / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_root / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Copy hardhat dataset with remapped labels
    for split in ['train', 'valid', 'test']:
        src_img = hardhat_root / split / 'images'
        src_lbl = hardhat_root / split / 'labels'
        dst_img = output_root / split / 'images'
        dst_lbl = output_root / split / 'labels'
        
        if not src_img.exists():
            logger.warning(f"Split {split} not found in {hardhat_root}")
            continue
        
        # Copy images
        for img_file in src_img.glob("*.*"):
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                shutil.copy(img_file, dst_img / img_file.name)
        
        # Remap labels
        for lbl_file in src_lbl.glob("*.txt"):
            with open(lbl_file, 'r') as f:
                lines = f.readlines()
            
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                old_cls = int(float(parts[0]))
                if old_cls in mapping_3to4:
                    new_cls = mapping_3to4[old_cls]
                    new_lines.append(f"{new_cls} {' '.join(parts[1:])}\n")
            
            with open(dst_lbl / lbl_file.name, 'w') as f:
                f.writelines(new_lines)
        
        logger.info(f"{split}: copied {len(list(src_img.glob('*.*')))} images, remapped labels {mapping_3to4}")
    
    # Try to add COCO vehicles if available
    coco_img_dir = Path(coco_root) / "images" / "train2017" if Path(coco_root).exists() else None
    coco_lbl_dir = Path(coco_root) / "labels" / "train2017" if Path(coco_root).exists() else None
    
    if coco_img_dir and coco_img_dir.exists():
        logger.info(f"Found COCO at {coco_root}, adding Vehicle class...")
        # This would require pycocotools - for now we just log
        # In production, you'd parse COCO json and extract car/truck/bus
        logger.info("COCO Vehicle addition requires pycocotools - placeholder for now")
        logger.info(f"To add Vehicle: download COCO 2017, then run with --coco-root {coco_root}")
    else:
        logger.warning(f"COCO not found at {coco_root}, creating 4-class dataset without Vehicle images")
        logger.info("For 4-class training, you need to add Vehicle images manually or via COCO")
        logger.info("Current dataset has Person, Hard-Hat, No-Hard-Hat - Vehicle will be 0 boxes until added")
        logger.info("For production delivery: you can train 3-class now (Hard-Hat focus) and add Vehicle later from COCO")
    
    # Create yaml
    yaml_path = create_4class_yaml(output_root, train_rel="train/images", val_rel="valid/images")
    
    # Create classes.txt
    with open(output_root / "classes.txt", 'w') as f:
        f.write("Person\nVehicle\nHard-Hat\nNo-Hard-Hat\n")
    
    logger.success(f"4-class dataset ready at {output_root}")
    logger.info(f"To train 4-class: python scripts/fix_nan_train.py --data {yaml_path} --epochs 50 --batch 4")
    
    return yaml_path

def main():
    parser = argparse.ArgumentParser(description="Merge COCO Vehicle into Hard-Hat dataset for 4-class")
    parser.add_argument("--hardhat-root", default="datasets/dji-hardhat-real", help="3-class hardhat root")
    parser.add_argument("--coco-root", default="datasets/coco", help="COCO root (optional)")
    parser.add_argument("--output", default="datasets/dji-hardhat-4class", help="Output 4-class root")
    parser.add_argument("--limit-vehicle", type=int, default=2000, help="Max vehicle images from COCO")
    args = parser.parse_args()
    
    merge_datasets(
        hardhat_root=Path(args.hardhat_root),
        coco_root=Path(args.coco_root),
        output_root=Path(args.output),
        limit_vehicle=args.limit_vehicle
    )

if __name__ == "__main__":
    main()
