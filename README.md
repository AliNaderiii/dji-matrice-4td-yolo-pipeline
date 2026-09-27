# DJI Matrice 4TD - Custom YOLO Deployment Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8%20%7C%20v11-green)](https://docs.ultralytics.com/)
[![ONNX](https://img.shields.io/badge/ONNX-Compatible-005CED)](https://onnx.ai/)
[![DJI](https://img.shields.io/badge/DJI-Matrice%204TD%20%7C%20Dock%203-red)](https://developer.dji.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Production-ready, reproducible pipeline to train, export, and deploy custom YOLO models to DJI Matrice 4TD NPU via DJI AI Open Platform, with MQTT JSON inference to AWS IoT Core @3fps.**

## 🎯 What This Solves

- **Problem:** DJI Matrice 4TD NPU requires INT8 quantized models in `.dji` format, deployed via Pilot 2 / FlightHub 2. Documentation is fragmented.
- **Solution:** End-to-end pipeline: Dataset → YOLOv8n training → DJI-compatible ONNX → DJI Portal quantization → MQTT JSON @ 3fps to AWS IoT Core.
- **Goal:** Reproducible workflow that any engineering team can own independently.

## 📊 Real Training Results - Honest Status

| Metric | Value | Notes |
|--------|-------|-------|
| Dataset | 4,916 train / 1,413 val | Person, Vehicle, Hard-Hat, No-Hard-Hat (intended) |
| Model | YOLOv8n 3.2M params | NPU optimized |
| Current Checkpoint | `runs/dji/yolov8n_4class/weights/best.pt` 5.6MB | 5 epochs CPU - **mAP50 0.0 - FAILED** |
| Previous Logs Claim | mAP50 0.655 Hard-Hat 0.981 | Not reproducible with current checkpoint |
| Status | **Retraining needed** | UAV pipeline shows real inference working (see other repo) |

> **Honest Note:** Current DJI checkpoint gives 0 detections even at conf 0.01 (verified). Training log shows 5 epochs CPU with mAP50 0.0. This is a failed run. UAV repo `uav-aerial-detection-yolo-pipeline` has 100% real inference (18-58 detections) proving pipeline works. DJI retraining pending with same pipeline.

## 📸 Demo - Status

**Current:** No real detection images - checkpoint gives 0 detections.

**Why:** Training was 5 epochs on CPU (see `runs/dji/yolov8n_4class/results.csv` - mAP50 0 all epochs).

**Next:**
```bash
# Proper training 50 epochs GPU
yolo detect train data=datasets/dji-hardhat/dji_matrice.yaml model=yolov8n.pt epochs=50 imgsz=640 batch=8 device=0 workers=0 amp=False optimizer=SGD lr0=0.001 project=runs/dji name=yolov8n_4class_real50

# Real inference after retrain
python generate_real_demo.py --model runs/dji/yolov8n_4class_real50/weights/best.pt --source datasets/dji-hardhat/images/val --output demo/real --num 5 --conf 0.25
```

**For now, see UAV repo for real inference example:** https://github.com/AliNaderiii/uav-aerial-detection-yolo-pipeline - 5 real images with 18-58 detections.

### Demo Files (Currently Only JSON)

- `demo/sample_payload.json` - MQTT JSON example @3fps
- `demo/dji_inference_log_sample.jsonl` - Sample inference log

## 🏗️ Architecture

```
[Dataset: Person, Vehicle, Hard-Hat, No-Hard-Hat]
        │
        ▼
[Ultralytics YOLOv8n Training]  (3.2M params, optimized for NPU)
        │  best.pt 5.6MB
        ▼
[ONNX Exporter]  (opset=12, static 640, simplify=True)
        │  best.onnx 10.5MB + calibration_images/ 150
        ▼
[DJI AI Developer Portal]  ai-developer.dji.com
        │  Upload .onnx/.pth + calibration → INT8 Quantization → .dji package 3-5MB
        ▼
[Matrice 4TD + Pilot 2 / FlightHub 2]  Bind to SN, Deploy
        │
        ▼
[Dock 3 + Cloud API]  JSON inference @ 3fps (mock publisher simulates this locally)
        │
        ▼
[AWS IoT Core]  MQTT Topic: dji/matrice4td/inference → Backend
```

## 📁 Repository Structure

```
├── configs/                # Dataset & model configs
├── src/                    # Production source code
│   ├── dataset/            # Downloader, validator, calibration set
│   ├── training/           # Trainer wrapper with drone-specific augmentations
│   ├── export/             # ONNX exporter + validator
│   ├── deployment/         # DJI submission package generator
│   └── mqtt/               # AWS subscriber, DJI mock publisher, Pydantic models
├── notebooks/
│   └── 01_DJI_Matrice_4TD_End_to_End_Pipeline.ipynb  # MAIN
├── scripts/                # CLI entry points
├── docker/
│   └── docker-compose.yml  # EMQX broker for local testing
├── docs/
│   ├── SOP.md
│   ├── DJI_PORTAL_WALKTHROUGH.md
│   └── MQTT_SETUP.md
└── demo/
    ├── sample_payload.json
    └── dji_inference_log_sample.jsonl
```

## 🚀 Quick Start

```bash
git clone https://github.com/AliNaderiii/dji-matrice-4td-yolo-pipeline.git
cd dji-matrice-4td-yolo-pipeline
py -3.11 -m venv D:\\common-venv
D:\\common-venv\\Scripts\\Activate.ps1
pip install -r requirements.txt

# 1. Prepare dataset
python scripts/download_dataset.py --source roboflow --limit 2000

# 2. Train YOLOv8n (GTX 1650 stable) - 50 epochs needed, not 5
yolo detect train data=configs/dji_matrice.yaml model=yolov8n.pt epochs=50 imgsz=640 batch=4 workers=0 amp=False optimizer=SGD lr0=0.001 device=0

# 3. Export DJI-compatible ONNX
yolo export model=runs/detect/train/weights/best.pt format=onnx opset=12 imgsz=640 simplify=True

# 4. Create DJI submission package
python scripts/create_submission.py --onnx runs/detect/train/weights/best.onnx --calib-size 150

# 5. Test MQTT locally (requires EMQX)
docker-compose -f docker/docker-compose.yml up emqx -d
# Terminal 1: python scripts/run_mqtt_test.py --mode subscriber
# Terminal 2: python scripts/run_mqtt_test.py --mode mock --fps 3 --frames 100
```

## 📡 MQTT Integration @3fps

**Mock publisher simulates DJI Cloud API JSON output without hardware. Same schema as real DJI, compatible with AWS IoT Core.**

```python
from src.mqtt.aws_subscriber import DJIInferenceSubscriber

subscriber = DJIInferenceSubscriber(
    endpoint="your-endpoint-ats.iot.us-east-1.amazonaws.com",
    topic="dji/matrice4td/inference",
    use_tls=True
)
subscriber.start()
```

## 📊 Benchmark - ONNX Latency (Measured on UAV model)

| Model | Format | Size | CPU | GPU GTX 1650 | FPS GPU |
|-------|--------|------|-----|--------------|---------|
| YOLOv8n 4-class | PyTorch .pt | 5.6 MB | 45ms | 5.1ms | ~196 |
| YOLOv8n 4-class | ONNX opset12 | 10.5 MB | 60ms | 8ms | ~125 |

> NPU on Matrice 4TD: **estimated** ~45ms (22 fps), throttled to 3fps JSON. Actual measurement pending hardware. Measured on UAV working model.

## 📦 Demo Files

- `sample_payload.json` - MQTT JSON example @3fps
- `dji_inference_log_sample.jsonl` - Sample log

**GitHub Topics to add:** `yolo yolov8 onnx edge-ai mqtt aws-iot drone dji computer-vision`

**Status:** No fake images - all previous synthetic visuals removed. Real inference pending retraining. See UAV repo for real working example.

## 👨‍💻 Author

**Ali Naderi** - M.Sc. Mechatronics, AI Researcher, Edge AI Engineer
- Published: Wiley Complexity 2025 (98.16% brain tumor classification)
- GitHub: [AliNaderiii](https://github.com/AliNaderiii)
- Location: Dublin, Ireland

## 📄 License

MIT License - You own all code. Commercial use allowed.
