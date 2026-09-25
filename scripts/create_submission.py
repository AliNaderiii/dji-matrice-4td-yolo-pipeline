#!/usr/bin/env python3
"""
CLI: Create DJI submission package
Author: Ali Naderi

Usage:
    python scripts/create_submission.py --onnx runs/dji/yolov8n_real_fixed/weights/best.onnx --yaml datasets/dji-hardhat-real/dji_matrice_3class.yaml
    python scripts/create_submission.py --onnx runs/dji/yolov8n_real_fixed/weights/best.onnx --yaml datasets/dji-hardhat-real/dji_matrice_3class.yaml --pt runs/dji/yolov8n_real_fixed/weights/best.pt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset.calibration import CalibrationSetGenerator
from src.deployment.package_generator import DJISubmissionPackage
from loguru import logger


def main():
    parser = argparse.ArgumentParser(description="Create DJI submission package")
    parser.add_argument("--onnx", required=True, help="Path to best.onnx")
    parser.add_argument("--yaml", default="datasets/dji-hardhat-real/dji_matrice_3class.yaml", help="Dataset YAML (3-class real)")
    parser.add_argument("--pt", default=None, help="Path to best.pt (optional, for reference)")
    parser.add_argument("--calib-dir", default="./dji_submission/calibration_images", help="Existing calib dir (optional)")
    parser.add_argument("--calib-size", type=int, default=150, help="Number of calibration images to generate")
    parser.add_argument("--version", default="v1.0-real", help="Model version")
    parser.add_argument("--generate-calib", action="store_true", help="Force regenerate calibration set")
    
    args = parser.parse_args()
    
    logger.info("=== DJI Submission Package ===")
    logger.info(f"ONNX: {args.onnx}")
    logger.info(f"YAML: {args.yaml}")
    if args.pt:
        logger.info(f"PT: {args.pt} (will be included as reference)")
    
    # Check files exist
    if not Path(args.onnx).exists():
        logger.error(f"ONNX not found: {args.onnx}")
        return
    if not Path(args.yaml).exists():
        logger.error(f"YAML not found: {args.yaml}, trying fallback")
        # Try fallback paths
        fallbacks = [
            "datasets/dji-hardhat-real/dji_matrice_3class.yaml",
            "datasets/dji-hardhat-real/data.yaml",
            "./datasets/dji-hardhat/dji_matrice.yaml",
            "configs/dji_matrice.yaml"
        ]
        for fb in fallbacks:
            if Path(fb).exists():
                args.yaml = fb
                logger.info(f"Using fallback YAML: {fb}")
                break
    
    # Generate calibration set if needed or forced
    calib_dir = Path(args.calib_dir)
    if args.generate_calib or not calib_dir.exists() or len(list(calib_dir.glob("*"))) < 10:
        logger.info(f"Generating calibration set: {args.calib_size} images from {args.yaml}")
        try:
            generator = CalibrationSetGenerator(dataset_yaml=args.yaml, output_dir=args.calib_dir)
            calib_path = generator.generate(num_images=args.calib_size, strategy="diverse")
        except Exception as e:
            logger.warning(f"Calibration generation failed: {e}, using existing or empty")
            calib_path = str(calib_dir)
            Path(calib_path).mkdir(parents=True, exist_ok=True)
    else:
        calib_path = str(calib_dir)
        logger.info(f"Using existing calibration set: {calib_path} ({len(list(calib_dir.glob('*')))} images)")
    
    # Create package
    packager = DJISubmissionPackage(output_dir="./dji_submission")
    zip_path = packager.create_package(
        onnx_path=args.onnx,
        dataset_yaml=args.yaml,
        calibration_dir=calib_path,
        model_version=args.version,
        extra_notes="Construction safety monitoring: Person (2.4%), Hard-Hat (73%), No-Hard-Hat (24%). Optimized for Matrice 4TD NPU. mAP50 0.655, Hard-Hat 0.981, No-Hard-Hat 0.954. Trained on 4916 images, 50 epochs, SGD lr0 0.001, amp False for GTX 1650 stability. ONNX opset 12, 10.47MB.",
        pt_path=args.pt
    )
    
    logger.success(f"Submission package ready: {zip_path}")
    print(f"\nNext steps:")
    print(f"1. Upload {zip_path} to https://developer.dji.com/ai-developer/")
    print(f"2. Follow docs/DJI_PORTAL_WALKTHROUGH.md")
    print(f"3. Test MQTT: python scripts/run_mqtt_test.py --mode mock")


if __name__ == "__main__":
    main()
