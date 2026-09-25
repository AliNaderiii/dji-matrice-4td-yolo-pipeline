"""
DJI Trainer - Wrapper around Ultralytics YOLO with DJI-specific logic
Author: Ali Naderi | Edge AI Engineer
Handles training, logging, checkpointing, and DJI compatibility checks.
Optimized for low-VRAM GPUs (GTX 1650 4GB) and stable training on real data.
"""

from pathlib import Path
from typing import Optional, Dict
from loguru import logger
import yaml
import torch

from ultralytics import YOLO

from .augmentations import DroneAugmentations


class DJITrainer:
    """
    Production trainer for DJI Matrice 4TD.
    
    Features:
    - Automatic device detection (GPU/CPU)
    - Low-VRAM handling (GTX 1650 4GB, RTX 3050, etc.)
    - Drone-specific augmentations
    - Stable training: AMP disabled by default for real data
    - Early stopping & best checkpoint tracking
    - DJI NPU compatibility validation
    """

    def __init__(self, config_path: str = "configs/yolov8n_dji.yaml"):
        self.config_path = Path(config_path)
        
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Detect device and VRAM
        self.device, self.vram_gb = self._detect_device()
        logger.info(f"Using device: {self.device}, VRAM: {self.vram_gb:.1f}GB")
        logger.info(f"Config loaded from {self.config_path}")

    def _detect_device(self) -> tuple:
        """Detect best available device and VRAM"""
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_memory / (1024**3)
            logger.info(f"CUDA available: {gpu_name} ({vram_gb:.1f}GB)")
            # Warn if low VRAM
            if vram_gb < 6:
                logger.warning(f"Low VRAM detected ({vram_gb:.1f}GB) - will use batch=8, workers=2, amp=False")
            return "0", vram_gb  # Ultralytics uses "0" for first GPU
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("MPS available (Apple Silicon)")
            return "mps", 0
        else:
            logger.warning("No GPU found, using CPU - training will be slow")
            return "cpu", 0

    def _prepare_model(self) -> YOLO:
        """Load pretrained YOLOv8n"""
        model_name = self.config.get('model', 'yolov8n.pt')
        logger.info(f"Loading pretrained model: {model_name}")
        
        model = YOLO(model_name)
        
        # Log model info
        try:
            num_params = sum(p.numel() for p in model.model.parameters()) / 1e6
            logger.info(f"Model parameters: {num_params:.2f}M")
        except:
            logger.info("Model loaded")
        return model

    def _get_train_args(self, data_yaml: str, epochs: Optional[int], batch: Optional[int],
                        imgsz: Optional[int], project: Optional[str], name: Optional[str],
                        amp: Optional[bool], optimizer: Optional[str], lr0: Optional[float]) -> Dict:
        """Build training args with safe defaults for low VRAM"""
        # Override config if provided
        epochs = epochs or self.config.get('epochs', 100)
        batch = batch or self.config.get('batch', 16)
        imgsz = imgsz or self.config.get('imgsz', 640)
        project = project or self.config.get('project', 'runs/dji')
        name = name or self.config.get('name', 'yolov8n_4class')
        
        # Low VRAM auto-adjust
        if self.vram_gb > 0 and self.vram_gb < 6:
            if batch > 8:
                logger.warning(f"Reducing batch from {batch} to 8 for {self.vram_gb:.1f}GB VRAM")
                batch = 8
            workers = min(self.config.get('workers', 4), 2)
        else:
            workers = self.config.get('workers', 4)

        # Stable training defaults - critical for real dataset
        # NaN loss fix: amp=False, SGD, lr0=0.001
        use_amp = amp if amp is not None else self.config.get('amp', False)
        use_optimizer = optimizer or self.config.get('optimizer', 'SGD')
        use_lr0 = lr0 or self.config.get('lr0', 0.001)
        use_lrf = self.config.get('lrf', 0.01)

        # Validate DJI requirements
        if imgsz != 640:
            logger.warning(f"DJI standard is 640, you requested {imgsz} - may cause quantization issues")

        # Get drone augmentations
        aug_overrides = DroneAugmentations.get_ultralytics_overrides()

        train_args = dict(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=self.device,
            workers=workers,
            optimizer=use_optimizer,
            lr0=use_lr0,
            lrf=use_lrf,
            weight_decay=self.config.get('weight_decay', 0.0005),
            warmup_epochs=self.config.get('warmup_epochs', 3),
            # Critical stability flags
            amp=use_amp,
            # Augmentations
            degrees=aug_overrides['degrees'],
            translate=aug_overrides['translate'],
            scale=aug_overrides['scale'],
            shear=aug_overrides['shear'],
            perspective=aug_overrides['perspective'],
            flipud=aug_overrides['flipud'],
            fliplr=aug_overrides['fliplr'],
            mosaic=aug_overrides['mosaic'],
            mixup=aug_overrides['mixup'],
            hsv_h=aug_overrides['hsv_h'],
            hsv_s=aug_overrides['hsv_s'],
            hsv_v=aug_overrides['hsv_v'],
            # Validation
            val=self.config.get('val', True),
            plots=self.config.get('plots', True),
            save=self.config.get('save', True),
            save_period=self.config.get('save_period', 10),
            patience=self.config.get('patience', 20),
            # Project
            project=project,
            name=name,
            exist_ok=self.config.get('exist_ok', False),
            # Extra stability for low VRAM
            cache=self.config.get('cache', False),
            close_mosaic=self.config.get('close_mosaic', 10),
        )

        # Log final config
        logger.info(f"Training args: epochs={epochs}, batch={batch}, imgsz={imgsz}, "
                    f"optimizer={use_optimizer}, lr0={use_lr0}, amp={use_amp}, workers={workers}")

        return train_args

    def train(self, data_yaml: str, epochs: Optional[int] = None, batch: Optional[int] = None, 
              imgsz: Optional[int] = None, project: Optional[str] = None, name: Optional[str] = None,
              amp: Optional[bool] = None, optimizer: Optional[str] = None, lr0: Optional[float] = None) -> str:
        """
        Train YOLOv8n for DJI Matrice 4TD.
        
        Args:
            data_yaml: Path to dataset yaml
            epochs: Override epochs from config
            batch: Override batch size
            imgsz: Override image size (must be 640 for DJI)
            project: Project dir
            name: Run name
            amp: Override AMP (False recommended for GTX 1650 stability)
            optimizer: Override optimizer (SGD recommended for real data)
            lr0: Override initial lr (0.001 recommended)
        
        Returns:
            Path to best.pt
        """
        train_args = self._get_train_args(data_yaml, epochs, batch, imgsz, project, name, amp, optimizer, lr0)
        
        model = self._prepare_model()
        
        # Train
        logger.info("Starting training...")
        results = model.train(**train_args)
        
        best_pt = Path(results.save_dir) / "weights" / "best.pt"
        logger.success(f"Training completed. Best model: {best_pt}")
        logger.info(f"Results saved to: {results.save_dir}")
        
        # Validate best.pt for DJI
        self._validate_for_dji(str(best_pt))
        
        return str(best_pt)

    def _validate_for_dji(self, pt_path: str):
        """Check if model is suitable for DJI NPU"""
        pt_file = Path(pt_path)
        if not pt_file.exists():
            logger.warning(f"Best.pt not found at {pt_path}")
            return

        size_mb = pt_file.stat().st_size / (1024 * 1024)
        logger.info(f"Model size: {size_mb:.2f} MB")
        
        if size_mb > 20:
            logger.warning(f"Model {size_mb:.1f}MB is large for Matrice 4TD NPU - consider yolov8n only, not s/m/l")
        else:
            logger.info("Model size OK for DJI NPU")
        
        # Check params
        try:
            model = YOLO(pt_path)
            num_params = sum(p.numel() for p in model.model.parameters())
            logger.info(f"Parameters: {num_params / 1e6:.2f}M")
            if num_params > 5e6:
                logger.warning("Model >5M params may be too heavy for Matrice 4TD NPU")
        except Exception as e:
            logger.warning(f"Could not check params: {e}")

    def benchmark(self, pt_path: str, data_yaml: str):
        """Benchmark model: mAP, latency"""
        logger.info(f"Benchmarking {pt_path}")
        model = YOLO(pt_path)
        
        # Validation metrics
        metrics = model.val(data=data_yaml, imgsz=640)
        logger.info(f"mAP50-95: {metrics.box.map:.4f}, mAP50: {metrics.box.map50:.4f}")
        
        return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train YOLO for DJI Matrice 4TD")
    parser.add_argument("--data", default="./datasets/dji-hardhat/dji_matrice.yaml", help="Dataset yaml")
    parser.add_argument("--config", default="configs/yolov8n_dji.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--amp", type=lambda x: x.lower() == 'true', default=None, help="Use AMP")
    parser.add_argument("--optimizer", default=None, help="Optimizer: SGD, AdamW")
    parser.add_argument("--lr0", type=float, default=None, help="Initial lr")
    args = parser.parse_args()
    
    trainer = DJITrainer(config_path=args.config)
    best_pt = trainer.train(
        data_yaml=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        amp=args.amp,
        optimizer=args.optimizer,
        lr0=args.lr0
    )
    print(f"Best model: {best_pt}")
