# DJI AI Open Platform - Detailed Walkthrough

## Source: Official DJI docs + RIIS tutorial + Reddit community experiences

### Step 0: Understand DJI's New AI Platform

DJI Matrice 4TD/4E/4D series have built-in NPU (Neural Processing Unit).
Unlike old Matrice 300 which needed Manifold, 4TD runs models on-board.

Platform: https://developer.dji.com/ai-developer/ (also called AI Inside)

### Step 1: Apply as Algorithm Developer

1. Go to https://developer.dji.com/ → Register with company email
2. Go to https://developer.dji.com/ai-developer/ → Apply
3. Fill:
   - Company: Your company
   - Use case: "Construction site safety monitoring - detection of Person, Vehicle, Hard-Hat, No-Hard-Hat for automated safety alerts via Dock 3"
   - Platform: Matrice 4TD
   - Model: YOLOv8n custom
   - Expected deployment: Dock 3 autonomous
4. Wait 1-2 business days. DJI will email approval.

**Tip:** Use company domain email, not gmail, for faster approval.

### Step 2: Create Project in Portal

After approval:
- Login to AI Developer Portal
- Click "New Project"
- Name: `Safety-Monitoring-v1`
- Platform: Matrice 4TD (or Matrice 4 Series)
- Task Type: Object Detection
- Description: Construction safety

### Step 3: Prepare Model for Upload

**Option A: ONNX (Recommended for 4TD - new platform)**

From our pipeline:
```bash
python scripts/export.py --weights runs/dji/yolov8n_4class/weights/best.pt --opset 12
python scripts/create_submission.py --onnx runs/dji/.../best.onnx --calib-size 150
```

You get `dji_submission/dji_matrice_4td_v1.0.zip` containing:
- best.onnx
- classes.txt
- calibration_images/ (150 images)
- calibration_list.txt
- metadata.json

**Option B: PyTorch .pth (Legacy, for Matrice 4E, fallback)**

If portal rejects ONNX and asks for .pth:

1. Install MMYOLO:
```bash
pip install openmim
mim install mmyolo
```

2. Use config: `configs/yolov8/yolov8_n_syncbn_fast_8xb16-500e_coco.py`
   Modify:
   - num_classes=4
   - dataset to your dji_matrice.yaml

3. Train with MMYOLO:
```bash
python tools/train.py configs/yolov8/yolov8_n_syncbn_fast_8xb16-500e_dji.py
```

4. Output: `work_dirs/.../best_coco_bbox_mAP_epoch_*.pth`

5. Submit: .pth + calibration_images + config

Reference: https://www.riis.com/blog/deploying-custom-ml-models-on-the-dji-matrice-4e

### Step 4: Upload & Quantization

In portal:
- Click "Upload Model"
- Select zip or onnx + calibration
- Upload
- Click "Start Quantization"

**What happens server-side:**
- DJI converts ONNX/PTH to INT8 using calibration images (PTQ)
- Optimizes for Matrice 4TD NPU
- Generates .dji package
- Takes 10-30 mins
- You can see logs

**Check logs for:**
- "Quantization successful"
- "Accuracy: mAP50 0.89 -> 0.88 (drop 1.1%)" - should be <2%
- If drop >5%, calibration set not diverse enough - regenerate

### Step 5: Bind to Device

1. Go to "Device Management"
2. Click "Add Device"
3. Enter Matrice 4TD SN (found in Pilot 2 → About → SN, format like 1581F6Q8D245P00EKS87)
4. Assign model to device
5. Click "Authorize" - DJI binds model to SN (license)

**Important:** Model is bound to SN - can't be used on other drones without re-binding.

### Step 6: Deploy via Pilot 2

On RC Plus 2 controller:

1. Connect to internet (WiFi)
2. Open DJI Pilot 2
3. Go to "AI Model Management" or "Algorithm Management"
4. Click "Sync" - your model should appear
5. Click "Download" to drone
6. Enable model: Set confidence 0.5, NMS 0.45, select classes
7. Go to camera view - you should see bounding boxes overlay

**Test:** Point drone at person with/without hard-hat, check detection.

### Step 7: FlightHub 2 & Dock 3 (For Autonomous)

If using Dock 3:

1. In FlightHub 2 → Dock → Select your Dock
2. Go to "AI Settings" → Enable custom model
3. Select your deployed model
4. Set output: JSON, 3fps
5. Configure MQTT broker (AWS IoT endpoint) in Cloud API settings

### Troubleshooting

See TROUBLESHOOTING.md for detailed errors.

Common:
- Quantization fails → check opset, try 11, check SiLU op
- Model not showing in Pilot 2 → check SN binding, internet, Pilot 2 version
- Low accuracy after quant → regenerate calibration with more diverse images
