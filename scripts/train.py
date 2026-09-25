#!/usr/bin/env python3
"""
CLI: Train YOLOv8n for DJI Matrice 4TD
Author: Ali Naderi | Edge AI Engineer

Usage:
    python scripts/train.py --data datasets/dji-hardhat-real/dji_matrice_3class.yaml --epochs 50 --batch 8
    python scripts/train.py --data datasets/dji-hardhat/dji_matrice.yaml --epochs 100 --batch 16
    # For GTX 1650 4GB stability fix:
    python scripts/train.py --data datasets/dji-hardhat-real/dji_matrice_3class.yaml --epochs 50 --batch 8 --optimizer SGD --lr0 0.001 --no-amp
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.trainer import DJITrainer
from loguru import logger


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8n for DJI Matrice 4TD")
    parser.add_argument("--data", default="./datasets/dji-hardhat/dji_matrice.yaml", help="Dataset YAML")
    parser.add_argument("--config", default="configs/yolov8n_dji.yaml", help="Training config YAML")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch", type=int, default=None, help="Override batch size")
    parser.add_argument("--imgsz", type=int, default=None, help="Override image size (must be 640 for DJI)")
    parser.add_argument("--project", default=None, help="Project dir")
    parser.add_argument("--name", default=None, help="Run name")
    parser.add_argument("--optimizer", default=None, choices=["SGD", "Adam", "AdamW"], help="Optimizer (SGD for stability)")
    parser.add_argument("--lr0", type=float, default=None, help="Initial learning rate (0.001 for stability)")
    parser.add_argument("--amp", dest="amp", action="store_true", help="Enable AMP (not recommended for 4GB)")
    parser.add_argument("--no-amp", dest="amp", action="store_false", help="Disable AMP (recommended for GTX 1650)")
    parser.set_defaults(amp=None)
    
    args = parser.parse_args()
    
    logger.info("=== DJI Matrice 4TD Training ===")
    logger.info(f"Data: {args.data}")
    logger.info(f"Config: {args.config}")
    if args.amp is not None:
        logger.info(f"AMP override: {args.amp}")
    
    trainer = DJITrainer(config_path=args.config)
    best_pt = trainer.train(
        data_yaml=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        project=args.project,
        name=args.name,
        amp=args.amp,
        optimizer=args.optimizer,
        lr0=args.lr0
    )
    
    logger.success(f"Training finished. Best model: {best_pt}")
    print(f"\nNext step: python scripts/export.py --weights {best_pt}")


if __name__ == "__main__":
    main()
