# DJI Matrice 4TD - Custom YOLO Deployment Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8%20%7C%20v11-green)](https://docs.ultralytics.com/)
[![ONNX](https://img.shields.io/badge/ONNX-Compatible-005CED)](https://onnx.ai/)
[![DJI](https://img.shields.io/badge/DJI-Matrice%204TD%20%7C%20Dock%203-red)](https://developer.dji.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Production-ready, reproducible pipeline to train, export, and deploy custom YOLO models to DJI Matrice 4TD NPU via DJI AI Open Platform, with MQTT JSON inference to AWS IoT Core.**

This repository provides a production-ready workflow for training and deploying custom YOLO models to edge devices.

---

## 🎯 What This Solves

- **Problem:** DJI Matrice 4TD NPU requires INT8 quantized models in `.dji` format, deployed via Pilot 2 / FlightHub 2. Documentation is fragmented and deployment is non-trivial.
- **Solution:** End-to-end pipeline: Dataset → YOLOv8n training → DJI-compatible ONNX → DJI Portal quantization → MQTT JSON @ 3fps to AWS IoT Core.
- **Goal:** Reproducible, well-documented workflow that any engineering team can own independently.

### Deliverables Included

- ✅ Reproducible training scripts + Google Colab notebook (you own the code)
- ✅ DJI-compatible ONNX exporter (opset 12, static 640x640, simplified)
- ✅ Calibration set generator for INT8 quantization (150 images)
- ✅ DJI AI Open Platform step-by-step guide + troubleshooting
- ✅ MQTT integration: Dock 3 Cloud API → AWS IoT Core (paho-mqtt, TLS, 3fps)
- ✅ SOP document + video script for internal team

---

## 🏗️ Architecture

```
[Dataset: Person, Vehicle, Hard-Hat, No-Hard-Hat]
        │
        ▼
[Ultralytics YOLOv8n Training]  (3.2M params, optimized for NPU)
        │  best.pt
        ▼
[ONNX Exporter]  (opset=12, dynamic=False, simplify=True)
        │  best.onnx (6-12MB FP32) + calibration_images/
        ▼
[DJI AI Developer Portal]  ai-developer.dji.com
        │  Upload .onnx/.pth + calibration → INT8 Quantization → .dji package
        ▼
[Matrice 4TD + Pilot 2 / FlightHub 2]  Bind to SN, Deploy
        │
        ▼
[Dock 3 + Cloud API]  JSON inference @ 3fps (not video stream)
        │
        ▼
[AWS IoT Core]  MQTT Topic: dji/matrice4td/inference → Your Backend
```

---

## 📁 Repository Structure

```
├── configs/                # Dataset & model configs
│   ├── dji_matrice.yaml    # 4-class dataset definition
│   └── yolov8n_dji.yaml    # Training hyperparams
├── src/                    # Production source code
│   ├── dataset/            # Downloader, validator, calibration set
│   ├── training/           # Trainer wrapper with drone-specific augmentations
│   ├── export/             # ONNX & MMYOLO exporters + validator
│   ├── deployment/         # DJI submission package generator
│   └── mqtt/               # AWS subscriber, DJI mock publisher, Pydantic models
├── notebooks/
│   └── 01_DJI_Matrice_4TD_End_to_End_Pipeline.ipynb  # MAIN - calls all scripts
├── scripts/                # CLI entry points
│   ├── train.py
│   ├── export.py
│   └── create_submission.py
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml  # EMQX broker for local testing
├── docs/
│   ├── SOP.md
│   ├── DJI_PORTAL_WALKTHROUGH.md
│   └── MQTT_SETUP.md
└── demo/
    └── sample_payload.json
```

---

## 🚀 Quick Start

### Option A: Google Colab (Recommended for Mentorship)

1. Open `notebooks/01_DJI_Matrice_4TD_End_to_End_Pipeline.ipynb` in Colab
2. Run all cells - it will install deps, download sample Hard-Hat dataset, train 10 epochs, export ONNX, and create DJI submission package.

### Option B: Local

```bash
# Clone & setup
git clone https://github.com/AliNaderiii/dji-matrice-4td-yolo-pipeline.git
cd dji-matrice-4td-yolo-pipeline
conda create -n dji python=3.10 -y && conda activate dji
pip install -r requirements.txt

# 1. Prepare dataset (downloads Hard-Hat Workers + validates)
python scripts/download_dataset.py --source roboflow --limit 2000

# 2. Train YOLOv8n
python scripts/train.py --data configs/dji_matrice.yaml --epochs 100 --imgsz 640 --batch 16

# 3. Export DJI-compatible ONNX
python scripts/export.py --weights runs/dji/yolov8n_4class/weights/best.pt --opset 12

# 4. Create DJI submission package
python scripts/create_submission.py --onnx runs/dji/yolov8n_4class/weights/best.onnx --calib-size 150

# 5. Test MQTT (local)
docker-compose -f docker/docker-compose.yml up -d
python scripts/run_mqtt_test.py --mode mock
```

---

## 📊 Model Specification

| Parameter | Value | Reason |
|-----------|-------|--------|
| Architecture | YOLOv8n | 3.2M params, fits Matrice 4TD NPU TOPS |
| Input | 640x640x3 | DJI standard |
| Classes | 4 | Person, Vehicle, Hard-Hat, No-Hard-Hat |
| Export | ONNX opset 12, static, simplified | DJI requirement |
| Size FP32 | 6-12 MB | |
| Size INT8 (after DJI quant) | 3-5 MB | NPU optimized |
| Inference | ~45ms on NPU | ~22 fps, throttled to 3fps JSON |

---

## 🔧 DJI AI Open Platform - Deployment Steps

Detailed walkthrough in `docs/DJI_PORTAL_WALKTHROUGH.md`. Summary:

1. **Apply as Algorithm Developer:** https://developer.dji.com/ai-developer/ (1-2 days approval)
2. **Create Project:** Platform Matrice 4TD, Task Object Detection
3. **Upload:** `dji_submission/package.zip` containing `best.onnx`, `classes.txt`, `calibration_images/`
4. **Quantization:** DJI server does INT8 PTQ, check logs for accuracy drop <2%
5. **Bind & Deploy:** Device Management → Add SN → Assign model → Sync in Pilot 2
6. **Validate:** Live view shows bounding boxes

**Common fixes documented in `docs/TROUBLESHOOTING.md`.**

---

## 📡 MQTT Integration

**Requirement:** Output JSON @ 3fps instead of video stream.

```python
from src.mqtt.aws_subscriber import DJIInferenceSubscriber

subscriber = DJIInferenceSubscriber(
    endpoint="your-endpoint-ats.iot.us-east-1.amazonaws.com",
    topic="dji/matrice4td/inference",
    use_tls=True
)
subscriber.start()  # Blocks, prints detections
```

Expected payload:
```json
{
  "timestamp": 1710000000000,
  "drone_sn": "4TD-XXXX",
  "detections": [
    {"class_name": "Person", "confidence": 0.92, "bbox": [100, 200, 150, 300]},
    {"class_name": "No-Hard-Hat", "confidence": 0.87, "bbox": [105, 205, 145, 295]}
  ]
}
```

Full AWS IoT Core setup in `docs/MQTT_SETUP.md`.

---

## 🎓 Mentorship & SOP

This repo is built for teaching:

- **Notebook 01** is fully commented, each cell explains *why* not just *how*
- **SOP.md** is a 15-page document your team can follow independently
- **Video script** included for recording internal training

After one pair-programming session, your team can:
- [ ] Train custom YOLOv8n on any new classes
- [ ] Export DJI-compatible ONNX and validate
- [ ] Submit to DJI portal and deploy via Pilot 2
- [ ] Configure Dock 3 to publish JSON to AWS

---

## 📈 Advanced Features

This repository includes production extras for robust deployment:

- **Pydantic payload validation** for MQTT JSON (type-safe)
- **ONNX inference benchmark** script (latency, FPS)
- **MMYOLO exporter** as fallback if DJI portal requires .pth
- **Dockerfile** for reproducible environment
- **Dataset validator** that checks YOLO format, missing labels, class imbalance
- **Augmentations for drone perspective** (perspective warp, top-down)
- **AWS IoT Rule** example to forward to S3/Lambda
- **Mock publisher/subscriber** for end-to-end testing without hardware

---

## 👨‍💻 Author

**Ali Naderi** - M.Sc. Mechatronics, AI Researcher, Edge AI Engineer
- Published: Wiley Complexity 2025 (98.16% brain tumor classification)
- GitHub: [AliNaderiii](https://github.com/AliNaderiii)
- Portfolio: Traffic Density YOLO11 Nano → ONNX 10.1MB
- Location: Dublin, Ireland

---

## 📄 License

MIT License - You own all code. Commercial use allowed.

---

## 🙏 Acknowledgments

- Ultralytics YOLOv8
- DJI AI Open Platform & Cloud API
- Roboflow Hard Hat Workers Dataset
