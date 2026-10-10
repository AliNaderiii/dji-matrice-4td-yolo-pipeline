"""Validate local PPE YOLO splits before training."""
from __future__ import annotations

import argparse
import json

from .config import load_config
from .data import validate_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    _, manifest = validate_dataset(load_config(args.config))
    print(json.dumps(manifest, indent=2))
