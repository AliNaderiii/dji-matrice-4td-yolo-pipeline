# SOP: Custom YOLO Deployment to DJI Matrice 4TD NPU

**Version:** 1.0 | **Author:** Ali Naderi - Edge AI Engineer | **Date:** 2026-09-24
**Purpose:** Enable backend team to independently train, compile, and deploy custom YOLO models

---

## 1. Architecture Overview

```
Dataset (Person, Vehicle, Hard-Hat, No-Hard-Hat)
    ↓
YOLOv8n Training (3.2M params, 640x640)
    ↓ best.pt
ONNX Export (opset 12, static, simplified) → 6-12MB FP32
    ↓ + calibration_images/ (150 images)
DJI AI Developer Portal (ai-developer.dji.com)
    ↓ INT8 Quantization → .dji package (3-5MB)
Matrice 4TD + Pilot 2 / FlightHub 2 (Bind to SN, Deploy)
    ↓
Dock 3 + Cloud API → MQTT JSON @ 3fps
    ↓
AWS IoT Core → Backend
```

---

## 2. Phase 1: Pipeline Setup (Reproducible)

### 2.1 Environment

```bash
conda create -n dji python=3.10 -y
conda activate dji
pip install -r requirements.txt
```

For Colab: Open `notebooks/01_DJI_Matrice_4TD_End_to_End_Pipeline.ipynb` - it installs automatically.

### 2.2 Dataset Preparation

**Classes:**
- 0: Person (COCO)
- 1: Vehicle (COCO car, truck, bus mapped)
- 2: Hard-Hat (Roboflow Hard Hat Workers)
- 3: No-Hard-Hat (Roboflow)

**Steps:**

```bash
# Option A: Roboflow (real data, requires API key)
export ROBOFLOW_API_KEY=your_key
python scripts/download_dataset.py --source roboflow --limit 3000 --validate

# Option B: Synthetic demo (for pipeline testing, no API key)
python scripts/download_dataset.py --source synthetic --limit 500 --validate
```

**Output:** `datasets/dji-hardhat/` with `images/train|val`, `labels/train|val`, `dji_matrice.yaml`, `classes.txt`

**Validation:** Script checks YOLO format, class distribution, missing labels.

### 2.3 Training

```bash
python scripts/train.py --data datasets/dji-hardhat/dji_matrice.yaml --epochs 100 --batch 16
```

**Key Params for DJI NPU:**
- `model=yolov8n.pt` only (3.2M params) - s/m/l too heavy
- `imgsz=640` mandatory
- `degrees=0.0` - no rotation for drone horizon
- `scale=0.5` - altitude variation
- Early stopping patience 20

**Output:** `runs/dji/yolov8n_4class/weights/best.pt` + metrics, confusion matrix.

**Expected:** mAP50 85-95% depending on data quality. Check `results.png`.

### 2.4 ONNX Export (DJI Compatible)

```bash
python scripts/export.py --weights runs/dji/yolov8n_4class/weights/best.pt --opset 12 --benchmark
```

**DJI Requirements:**
- opset 11-14 (12 recommended)
- static shape [1,3,640,640] (dynamic=False)
- simplified graph
- FP32 (DJI quantizes to INT8 server-side)

**Validation:** Script runs `onnx.checker` and benchmarks latency.

**Output:** `best.onnx` (6-12MB) + `best_metadata.txt`

### 2.5 Calibration Set

```bash
python scripts/create_submission.py --onnx runs/dji/yolov8n_4class/weights/best.onnx --calib-size 150 --generate-calib
```

This generates `dji_submission/calibration_images/` with 150 diverse images covering all 4 classes (required for INT8 PTQ).

---

## 3. Phase 2: DJI AI Open Platform Deployment (Pair-Programming)

### 3.1 Prerequisites

1. DJI Developer Account: https://developer.dji.com/
2. Apply as Algorithm Developer: https://developer.dji.com/ai-developer/ → Apply → Fill company info (Construction Safety Monitoring), wait 1-2 days
3. Matrice 4TD Serial Number (Pilot 2 → About)
4. DJI RC Plus 2 controller with Pilot 2 updated

### 3.2 Portal Workflow (Live Call)

**Step 1: Create Project**
- Login → New Project → Name: `Safety-Monitoring-v1` → Platform: Matrice 4TD → Task: Object Detection

**Step 2: Upload Model**
- Option A (New 4TD): Upload `dji_submission/package.zip` containing `best.onnx`, `classes.txt`, `calibration_images/`, `metadata.json`
- Option B (Legacy 4E flow, fallback): Upload `best.pt` + config + calibration (see `docs/DJI_PORTAL_WALKTHROUGH.md`)

We try Option A first.

**Step 3: Quantization**
- DJI server quantizes to INT8 (10-30 mins)
- Check logs: "Quantization successful, accuracy drop <2%"
- If fails, see Troubleshooting below

**Step 4: Bind & Deploy**
- Device Management → Add Device → Enter SN
- Assign model to device → Download `.dji` package
- Open Pilot 2 → AI Model Management → Sync → Enable model, set conf 0.5, NMS 0.45
- Test live feed - bounding boxes should appear

**Step 5: Record**
- Record screen with OBS as `SOP_Video_Matrice4TD_Deployment.mp4` - deliverable.

### 3.3 Troubleshooting (Common)

| Error | Cause | Fix |
|-------|-------|-----|
| `Quantization failed: Unsupported op SiLU` | SiLU not supported in NPU | Export with `opset=11` or replace SiLU with ReLU in model (requires custom export) |
| `Input shape mismatch` | Dynamic axes or not 640 | Re-export with `dynamic=False, imgsz=640` |
| `Model too large` | Used yolov8s/m/l | Use yolov8n only, or `model.export(..., quantize=8)` for local PTQ test |
| `Calibration failed` | Not enough diverse images | Ensure 150 images covering all classes, different lighting |

Detailed in `docs/TROUBLESHOOTING.md`.

---

## 4. Phase 3: MQTT Integration (JSON @ 3fps)

### 4.1 Local Testing (Without Drone)

```bash
# Terminal 1: Start EMQX broker
docker-compose -f docker/docker-compose.yml up -d
# Dashboard: http://localhost:18083 admin/public

# Terminal 1: Start subscriber
python scripts/run_mqtt_test.py --mode subscriber

# Terminal 2: Start mock publisher @ 3fps
python scripts/run_mqtt_test.py --mode mock --frames 100 --fps 3.0
```

You should see JSON payloads @ 3fps in subscriber terminal, log saved to `demo/dji_inference_log.jsonl`.

### 4.2 AWS IoT Core (Production)

**1. AWS Setup:**
- IoT Core → Create Thing: `matrice-4td-01`
- Create certs: Download `AmazonRootCA1.pem`, `certificate.pem.crt`, `private.pem.key` to `./certs/`
- Create Policy: Allow `iot:Publish`, `iot:Subscribe`, `iot:Connect` on `dji/*`
- Attach policy to cert, cert to Thing
- Note endpoint: `xxxxxx-ats.iot.us-east-1.amazonaws.com`

**2. DJI FlightHub 2 / Dock 3 Config:**
- FlightHub 2 → Dock → Cloud API Settings → Custom MQTT Broker
- Endpoint: AWS endpoint, Port 8883, upload certs
- Topic: `dji/matrice4td/inference`
- AI Output: JSON, Frequency 3fps, Include bbox, conf, class

**3. Backend Subscriber:**

```bash
python scripts/run_mqtt_test.py --mode subscriber --endpoint your-endpoint-ats.iot.us-east-1.amazonaws.com --port 8883 --tls --cert-dir ./certs
```

**4. Validation:**
- Run 5 mins → ~900 messages (3fps * 300s)
- Show log `demo/dji_inference_log.jsonl` with all classes
- Screenshot AWS IoT MQTT test console showing messages
- This proves end-to-end pipeline

### 4.3 Expected JSON Payload

See `demo/sample_payload.json`:

```json
{
  "timestamp": 1710000000000,
  "drone_sn": "4TD-XXXX",
  "frame_id": 123,
  "detections": [
    {"class_name": "Person", "confidence": 0.92, "bbox": [100, 200, 150, 300]},
    {"class_name": "No-Hard-Hat", "confidence": 0.87, "bbox": [105, 205, 145, 295]}
  ],
  "inference_time_ms": 42.5
}
```

**Alert Logic:** If `No-Hard-Hat` conf >0.7 → Publish to `alerts/safety` or trigger Lambda.

Full guide in `docs/MQTT_SETUP.md`.

---

## 5. Checklist for Internal Team (After Mentorship)

After pair-programming call, team should be able to:

- [ ] Train YOLOv8n on custom 4-class dataset → best.pt
- [ ] Export ONNX with correct opset and validate with checker
- [ ] Create calibration set of 150 images
- [ ] Create DJI submission zip package
- [ ] Upload to DJI AI Portal, quantize, get .dji
- [ ] Bind to SN and deploy via Pilot 2
- [ ] Configure Dock 3 Cloud API to publish JSON @ 3fps
- [ ] Receive JSON in AWS IoT Core with provided subscriber
- [ ] Explain SOP to new engineer

---

## 6. Advanced Features Included

Production-grade extras:

- Pydantic payload validation (type-safe)
- ONNX inference benchmark
- MMYOLO fallback exporter
- Dockerfile + docker-compose for reproducible env
- Dataset validator (checks format, imbalance)
- Drone-specific augmentations (perspective, altitude)
- Safety violation alert logic
- Mock publisher for testing without hardware
- Comprehensive logging and error handling

---

## 7. Useful Links

- DJI AI Developer: https://developer.dji.com/ai-developer/
- DJI AI Inside: https://developer.dji.com/ai-inside/overview
- Ultralytics YOLOv8: https://docs.ultralytics.com/
- MMYOLO: https://github.com/open-mmlab/mmyolo
- DJI Cloud API: https://github.com/dji-sdk/Cloud-API-Doc
- Minimal Cloud API Example: https://github.com/pktiuk/DJI_Cloud_API_minimal
- Hard Hat Dataset: https://universe.roboflow.com/

---

**Next:** Schedule pair-programming call, share screen, walk through Phase 2 together.
