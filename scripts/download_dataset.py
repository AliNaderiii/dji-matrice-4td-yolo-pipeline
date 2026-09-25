#!/usr/bin/env python3
"""
CLI: Download dataset for DJI Matrice 4TD
Author: Ali Naderi

Usage:
    python scripts/download_dataset.py --source roboflow --limit 2000
    python scripts/download_dataset.py --source synthetic --limit 500
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset.downloader import DatasetDownloader
from src.dataset.validator import DatasetValidator
from loguru import logger


def main():
    parser = argparse.ArgumentParser(description="Download DJI dataset")
    parser.add_argument("--source", default="synthetic", choices=["roboflow", "synthetic"], help="Dataset source")
    parser.add_argument("--limit", type=int, default=2000, help="Max images")
    parser.add_argument("--config", default="configs/dji_matrice.yaml")
    parser.add_argument("--validate", action="store_true", help="Validate after download")
    
    args = parser.parse_args()
    
    logger.info("=== DJI Dataset Download ===")
    
    downloader = DatasetDownloader(config_path=args.config)
    yaml_path = downloader.run(source=args.source, limit=args.limit)
    
    logger.success(f"Dataset ready: {yaml_path}")
    
    if args.validate:
        validator = DatasetValidator(dataset_root=str(Path(yaml_path).parent))
        validator.run_full_validation(num_classes=4)


if __name__ == "__main__":
    main()
