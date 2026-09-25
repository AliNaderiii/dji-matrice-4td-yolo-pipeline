#!/usr/bin/env python3
"""
Fix for NaN + OOM + cuDNN HOST_ALLOCATION_FAILED on GTX 1650 4GB
Stable: SGD lr0 0.001 amp=False batch=4 workers=0

Usage:
    # Fresh start (recommended after OOM/cuDNN error):
    python scripts/fix_nan_train.py --data datasets/dji-hardhat-real/dji_matrice_3class.yaml --epochs 50 --batch 4 --workers 0
    
    # Resume (will force workers=0 by editing args.yaml if needed):
    python scripts/fix_nan_train.py --resume runs/dji/yolov8n_real_fixed/weights/last.pt --batch 4 --workers 0
"""
import argparse
from pathlib import Path
from loguru import logger
import yaml

def patch_args_yaml(run_dir: Path, batch: int, workers: int):
    """Patch args.yaml to force low RAM settings on resume"""
    args_yaml = run_dir / "args.yaml"
    if args_yaml.exists():
        try:
            with open(args_yaml, 'r') as f:
                args = yaml.safe_load(f)
            args['batch'] = batch
            args['workers'] = workers
            args['amp'] = False
            args['cache'] = False
            with open(args_yaml, 'w') as f:
                yaml.dump(args, f)
            logger.info(f"Patched {args_yaml}: batch={batch}, workers={workers}, amp=False")
        except Exception as e:
            logger.warning(f"Could not patch args.yaml: {e}")

def main():
    parser = argparse.ArgumentParser(description="Stable training for GTX 1650 4GB")
    parser.add_argument("--data", default="datasets/dji-hardhat-real/dji_matrice_3class.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=4, help="4 for 1650 4GB, 2 if still OOM")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--project", default="runs/dji")
    parser.add_argument("--name", default="yolov8n_real_fixed")
    parser.add_argument("--device", default="0")
    parser.add_argument("--resume", default=None, help="Path to last.pt")
    parser.add_argument("--workers", type=int, default=0, help="0 for low RAM")
    parser.add_argument("--fresh", action="store_true", help="Force fresh start even if last.pt exists")
    args = parser.parse_args()

    logger.info(f"=== GTX 1650 4GB Stable Training ===")
    logger.info(f"batch={args.batch}, workers={args.workers}, amp=False, SGD lr0=0.001")

    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed")
        return

    if args.resume and Path(args.resume).exists() and not args.fresh:
        run_dir = Path(args.resume).parent.parent
        patch_args_yaml(run_dir, args.batch, args.workers)
        logger.info(f"Resuming from {args.resume}")
        model = YOLO(args.resume)
        try:
            results = model.train(
                resume=True,
            )
        except Exception as e:
            logger.warning(f"Resume failed ({e}), trying with explicit batch/workers override via fresh model load")
            # Fallback: load last.pt as pretrained and continue
            model = YOLO(args.resume)
            results = model.train(
                data=args.data,
                epochs=args.epochs,
                batch=args.batch,
                imgsz=args.imgsz,
                device=args.device,
                workers=args.workers,
                optimizer="SGD",
                lr0=0.001,
                lrf=0.01,
                amp=False,
                cache=False,
                close_mosaic=10,
                degrees=0.0,
                translate=0.1,
                scale=0.5,
                shear=0.0,
                perspective=0.0005,
                flipud=0.0,
                fliplr=0.5,
                mosaic=1.0,
                mixup=0.0,
                hsv_h=0.015,
                hsv_s=0.7,
                hsv_v=0.4,
                val=True,
                plots=True,
                save=True,
                save_period=5,
                patience=15,
                project=args.project,
                name=args.name + "_resume",
                exist_ok=True,
            )
    else:
        model = YOLO("yolov8n.pt")
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            workers=args.workers,
            optimizer="SGD",
            lr0=0.001,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            amp=False,
            cache=False,
            close_mosaic=10,
            degrees=0.0,
            translate=0.1,
            scale=0.5,
            shear=0.0,
            perspective=0.0005,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.0,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            val=True,
            plots=True,
            save=True,
            save_period=5,
            patience=15,
            project=args.project,
            name=args.name,
            exist_ok=False,
        )

    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    logger.success(f"Training completed: {best_pt}")

if __name__ == "__main__":
    main()
