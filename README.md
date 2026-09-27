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

## 📊 Real Training Results

| Metric | Value | Notes |
|--------|-------|-------|
| Dataset | 4,916 train / 1,413 val | Person, Vehicle, Hard-Hat, No-Hard-Hat |
| Model | YOLOv8n 3.2M params, 6.9 GFLOPs | NPU optimized |
| Training | 50 epochs, GTX 1650 4GB stable | batch 4, workers 0, amp False, SGD lr0 0.001 |
| Overall | **mAP50 0.655** | Verified |
| Hard-Hat | **0.981** | Safety critical |
| No-Hard-Hat | **0.954** | Safety violation detection |
| ONNX | 10.5 MB, opset 12, static 640, simplified | DJI compatible |
| MQTT | 3 fps JSON telemetry | Mock publisher for local test, AWS IoT Core compatible |

> **NPU FPS Note:** ~45ms inference on NPU (~22 fps) is **estimated** based on 3.2M params and similar NPU benchmarks. Throttled to 3fps JSON for bandwidth. Actual NPU measurement pending hardware access. ONNX benchmark on GTX 1650: 5.1ms inference (~196 fps GPU).

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

## 🚀 Quick Start

```bash
git clone https://github.com/AliNaderiii/dji-matrice-4td-yolo-pipeline.git
cd dji-matrice-4td-yolo-pipeline
py -3.11 -m venv D:\common-venv
D:\common-venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 1. Prepare dataset
python scripts/download_dataset.py --source roboflow --limit 2000

# 2. Train YOLOv8n (GTX 1650 stable)
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

Expected payload:
```json
{
  "timestamp": 1710000000000,
  "drone_sn": "4TD-XXXX",
  "frame_id": 123,
  "detections": [
    {"class_name": "Person", "confidence": 0.92, "bbox": [100, 200, 150, 300]},
    {"class_name": "No-Hard-Hat", "confidence": 0.87, "bbox": [105, 205, 145, 295]}
  ],
  "fps": 3.0
}
```

Full setup in `docs/MQTT_SETUP.md`.

## 📊 Benchmark - ONNX Latency (Measured)

| Model | Format | Size | CPU | GPU GTX 1650 | FPS GPU |
|-------|--------|------|-----|--------------|---------|
| YOLOv8n 4-class | PyTorch .pt | 5.6 MB | 45ms | 5.1ms | ~196 |
| YOLOv8n 4-class | ONNX opset12 | 10.5 MB | 60ms | 8ms | ~125 |

> NPU on Matrice 4TD: **estimated** ~45ms (22 fps), throttled to 3fps JSON. Actual measurement pending hardware.

## 📸 Demo

Add your validation images with bounding boxes to `demo/`:

- `demo/sample_01.jpg` - Person + Hard-Hat detection
- `demo/sample_payload.json` - Example MQTT JSON

**TODO (2-hour checklist):**
- [ ] Add 2-3 detection samples from your val set to `demo/` and embed in README
- [ ] Add `runs/detect/train/confusion_matrix.png` to README
- [ ] Add GitHub topics: yolo, yolov8, onnx, edge-ai, mqtt, aws-iot, drone, dji, computer-vision

## License

MIT License - You own all code. Commercial use allowed.
