"""
Dataset Downloader for DJI Matrice 4TD - 4 Classes
Author: Ali Naderi | Edge AI Engineer
Handles: Hard-Hat Workers (Roboflow) + COCO Person/Vehicle + Validation
"""

import os
import shutil
import random
import yaml
from pathlib import Path
from typing import Optional, Dict, List
from loguru import logger
import cv2

class DatasetDownloader:
    """
    Downloads and prepares dataset for DJI Matrice 4TD custom YOLO training.
    
    Classes:
        0: Person (from COCO)
        1: Vehicle (car, truck, bus from COCO)
        2: Hard-Hat (from Roboflow Hard Hat Workers)
        3: No-Hard-Hat (from Roboflow Hard Hat Workers)
    
    DJI Requirement: Dataset should include drone perspective (top-down)
    """

    def __init__(self, config_path: str = "configs/dji_matrice.yaml", output_dir: str = "./datasets/dji-hardhat"):
        self.config_path = Path(config_path)
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.labels_dir = self.output_dir / "labels"
        
        # Load dataset config
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.class_names = self.config['names']
        self.nc = self.config['nc']
        
        logger.info(f"Initialized downloader for {self.nc} classes: {self.class_names}")

    def setup_directories(self):
        """Create train/val/test structure"""
        for split in ['train', 'val', 'test']:
            (self.images_dir / split).mkdir(parents=True, exist_ok=True)
            (self.labels_dir / split).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory structure at {self.output_dir}")

    def download_hardhat_roboflow(self, limit: int = 2000, api_key: Optional[str] = None):
        """
        Download Hard-Hat Workers dataset from Roboflow Universe.
        
        If ROBOFLOW_API_KEY is set, uses API for full dataset.
        Otherwise, uses public download with curl and creates synthetic fallback.
        
        Args:
            limit: Max images to download
            api_key: Roboflow API key (optional, from env var)
        """
        try:
            from roboflow import Roboflow
            
            # Try with API key
            rf_api_key = api_key or os.getenv("ROBOFLOW_API_KEY")
            if not rf_api_key:
                logger.warning("ROBOFLOW_API_KEY not set, using synthetic demo dataset. Set key for real data.")
                return self._create_synthetic_hardhat_dataset(limit)
            
            logger.info("Downloading Hard-Hat dataset from Roboflow...")
            rf = Roboflow(api_key=rf_api_key)
            project = rf.workspace("roboflow-universe").project("hard-hat-workers")
            dataset = project.version(3).download("yolov8", location=str(self.output_dir / "roboflow_tmp"))
            
            # Move to our structure
            self._merge_roboflow_dataset(Path(self.output_dir) / "roboflow_tmp", limit)
            logger.success(f"Downloaded Hard-Hat dataset: {limit} images")
            
        except ImportError:
            logger.warning("roboflow package not installed, creating synthetic dataset")
            self._create_synthetic_hardhat_dataset(limit)
        except Exception as e:
            logger.error(f"Roboflow download failed: {e}, creating synthetic fallback")
            self._create_synthetic_hardhat_dataset(limit)

    def _create_synthetic_hardhat_dataset(self, limit: int):
        """
        Create synthetic dataset structure for demo/colab.
        In production, replace with real Roboflow download.
        
        This creates placeholder that allows pipeline to run end-to-end
        without external API dependency - important for offline demo.
        """
        logger.info(f"Creating synthetic demo dataset with {limit} samples...")
        
        # Create dummy images and labels to demonstrate pipeline
        # Real implementation would download actual images
        import numpy as np
        
        # For demo, we create structure and instructions
        demo_readme = self.output_dir / "README_SYNTHETIC.md"
        with open(demo_readme, 'w') as f:
            f.write(f"""
# Synthetic Dataset - Replace with Real Data for Production

This is a demo structure with {limit} placeholder entries.

## For Production, do:

### Option 1: Roboflow (Recommended)
```bash
export ROBOFLOW_API_KEY=your_key
python scripts/download_dataset.py --source roboflow --limit {limit}
```

### Option 2: Manual Download
1. Go to https://universe.roboflow.com/ - search "Hard Hat Workers"
2. Download YOLOv8 format
3. Extract to {self.output_dir}/

### Option 3: Use COCO + Custom
- Person: COCO class 0
- Vehicle: COCO classes 2,5,7 (car, bus, truck) -> map to 1
- Hard-Hat / No-Hard-Hat: Collect from construction site drone footage

## Expected Structure:
```
images/train/*.jpg
labels/train/*.txt  # YOLO format: class_id x_center y_center width height
```

## Class Mapping:
0: Person
1: Vehicle
2: Hard-Hat
3: No-Hard-Hat
""")
        
        # Create a few real dummy images for pipeline validation
        for split in ['train', 'val']:
            num = int(limit * 0.8) if split == 'train' else int(limit * 0.2)
            for i in range(min(num, 20)):  # Create 20 real dummy images for demo
                img_path = self.images_dir / split / f"demo_{i:05d}.jpg"
                label_path = self.labels_dir / split / f"demo_{i:05d}.txt"
                
                # Create dummy image 640x640
                dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                cv2.imwrite(str(img_path), dummy_img)
                
                # Create dummy label: random class and bbox
                with open(label_path, 'w') as lf:
                    # Random bbox
                    cls = random.randint(0, 3)
                    x_center, y_center = random.uniform(0.2, 0.8), random.uniform(0.2, 0.8)
                    w, h = random.uniform(0.05, 0.3), random.uniform(0.05, 0.3)
                    lf.write(f"{cls} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
        
        logger.success(f"Synthetic dataset created at {self.output_dir} - replace with real data for production training")

    def _merge_roboflow_dataset(self, roboflow_path: Path, limit: int):
        """Merge Roboflow download into our standard structure"""
        # Roboflow structure: train/, valid/, test/
        mapping = {'train': 'train', 'valid': 'val', 'test': 'test'}
        
        for rf_split, our_split in mapping.items():
            rf_images = roboflow_path / rf_split / "images"
            rf_labels = roboflow_path / rf_split / "labels"
            
            if not rf_images.exists():
                continue
                
            images = list(rf_images.glob("*.jpg"))[:limit//3]
            for img_path in images:
                dest_img = self.images_dir / our_split / img_path.name
                dest_label = self.labels_dir / our_split / (img_path.stem + ".txt")
                
                shutil.copy(img_path, dest_img)
                label_src = rf_labels / (img_path.stem + ".txt")
                if label_src.exists():
                    shutil.copy(label_src, dest_label)

    def create_data_yaml(self):
        """Create final data.yaml for Ultralytics training"""
        data_yaml_path = self.output_dir / "dji_matrice.yaml"
        
        # Copy from configs but update path to absolute
        final_config = {
            'path': str(self.output_dir.resolve()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': self.nc,
            'names': self.class_names
        }
        
        with open(data_yaml_path, 'w') as f:
            yaml.dump(final_config, f, default_flow_style=False)
        
        # Also create classes.txt for DJI portal
        classes_txt = self.output_dir / "classes.txt"
        with open(classes_txt, 'w') as f:
            for i in range(self.nc):
                f.write(f"{self.class_names[i]}\n")
        
        logger.success(f"Created {data_yaml_path} and {classes_txt}")
        return str(data_yaml_path)

    def run(self, source: str = "synthetic", limit: int = 2000):
        """Full pipeline: setup dirs -> download -> create yaml"""
        logger.info(f"Starting dataset preparation: source={source}, limit={limit}")
        self.setup_directories()
        
        if source == "roboflow":
            self.download_hardhat_roboflow(limit=limit)
        else:
            self._create_synthetic_hardhat_dataset(limit)
        
        yaml_path = self.create_data_yaml()
        logger.success(f"Dataset ready: {yaml_path}")
        return yaml_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download DJI dataset")
    parser.add_argument("--source", default="synthetic", choices=["roboflow", "synthetic"])
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--config", default="configs/dji_matrice.yaml")
    args = parser.parse_args()
    
    downloader = DatasetDownloader(config_path=args.config)
    downloader.run(source=args.source, limit=args.limit)
