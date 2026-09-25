#!/usr/bin/env python3
"""
CLI: Export YOLO to DJI-compatible ONNX
Author: Ali Naderi

Usage:
    python scripts/export.py --weights runs/dji/yolov8n_4class/weights/best.pt --opset 12
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.export.onnx_exporter import DJIOnnxExporter
from src.export.validator import OnnxValidator
from loguru import logger


def main():
    parser = argparse.ArgumentParser(description="Export YOLO to DJI-compatible ONNX")
    parser.add_argument("--weights", required=True, help="Path to best.pt")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset (DJI recommends 11-14)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (must be 640 for DJI)")
    parser.add_argument("--output", default=None, help="Output dir")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark inference")
    
    args = parser.parse_args()
    
    logger.info("=== DJI ONNX Export ===")
    logger.info(f"Weights: {args.weights}")
    logger.info(f"Opset: {args.opset}, Imgsz: {args.imgsz}")
    
    exporter = DJIOnnxExporter(opset=args.opset, imgsz=args.imgsz)
    onnx_path = exporter.export(args.weights, output_dir=args.output, dynamic=False)
    
    logger.success(f"ONNX exported: {onnx_path}")
    
    if args.benchmark:
        validator = OnnxValidator()
        validator.benchmark_inference(onnx_path)
    
    print(f"\nNext step: python scripts/create_submission.py --onnx {onnx_path}")


if __name__ == "__main__":
    main()
