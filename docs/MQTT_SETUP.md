# MQTT Integration - Dock 3 + Cloud API + AWS IoT Core

## Requirement
> Show team how to configure drone/Dock 3 (via DJI Cloud API) to output inference results (JSON payload) over MQTT at ~3 fps, instead of streaming video.

## Architecture

```
Matrice 4TD NPU (YOLOv8n INT8)
    ↓ inference @ 22fps
Pilot 2 / Onboard (throttle to 3fps JSON)
    ↓ MQTT publish
DJI Cloud API (EMQX or AWS IoT)
    ↓ topic: dji/matrice4td/inference
AWS IoT Core
    ↓ Rule: SELECT * FROM 'dji/matrice4td/inference'
S3 / Lambda / Your Backend
```

Why JSON not video?
- Bandwidth: Video 5-10 Mbps, JSON 5-10 KBps (1000x savings)
- Privacy: No video stored, only metadata
- Battery: Less transmission

## Option 1: Local Testing with EMQX (No AWS, No Drone)

For development and deliverable validation:

```bash
# Start EMQX broker
docker-compose -f docker/docker-compose.yml up -d

# Check dashboard: http://localhost:18083 admin/public

# Terminal 1: Subscriber
python scripts/run_mqtt_test.py --mode subscriber --endpoint localhost --port 1883 --topic dji/matrice4td/inference

# Terminal 2: Mock publisher @ 3fps (simulates drone)
python scripts/run_mqtt_test.py --mode mock --endpoint localhost --port 1883 --topic dji/matrice4td/inference --fps 3.0 --frames 100
```

You should see:
- Subscriber prints detections with FPS ~3
- Log file `demo/dji_inference_log.jsonl` created
- Safety violations flagged

This proves end-to-end without hardware - perfect for SOP validation.

## Option 2: AWS IoT Core (Production)

### 2.1 AWS Setup

1. **Create Thing:**
   - IoT Core → Manage → Things → Create → Single Thing
   - Name: `matrice-4td-01`
   - No shadow

2. **Create Certificates:**
   - Auto-generate certs
   - Download:
     - `AmazonRootCA1.pem` (Root CA)
     - `certificate.pem.crt` (Device cert)
     - `private.pem.key` (Private key)
   - Save to `./certs/` (never commit to git!)

3. **Create Policy:**
   - IoT Core → Security → Policies → Create
   - Name: `matrice-4td-policy`
   - JSON:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["iot:Publish", "iot:Subscribe", "iot:Receive", "iot:Connect"],
      "Resource": ["*"]
    }
  ]
}
```
   - For production, restrict Resource to `arn:aws:iot:us-east-1:xxx:topic/dji/*`

4. **Attach:**
   - Policy → Attach to certificate
   - Certificate → Attach to Thing

5. **Get Endpoint:**
   - IoT Core → Settings → Endpoint: `xxxxxx-ats.iot.us-east-1.amazonaws.com`

### 2.2 DJI Cloud API Configuration

**If using DJI's Cloud API sample:**

```bash
git clone https://github.com/dji-sdk/Cloud-API-Doc
cd Cloud-API-Doc
# Follow README to setup Python env and configure .env with AWS endpoint
```

**If using FlightHub 2 + Dock 3:**

1. FlightHub 2 → Project → Dock → Select Dock 3
2. Settings → Cloud API → Custom MQTT Broker
   - Endpoint: `xxxxxx-ats.iot.us-east-1.amazonaws.com`
   - Port: 8883
   - Upload certs from `./certs/`
   - Topic: `dji/matrice4td/inference`
3. AI Settings:
   - Enable custom model: `Safety-Monitoring-v1`
   - Output type: JSON
   - Frequency: 3 fps
   - Include: bbox, confidence, class_id, class_name
   - Confidence threshold: 0.5

4. Save & Sync to Dock

### 2.3 Backend Subscriber

Use provided subscriber:

```bash
# Install
pip install paho-mqtt pydantic

# Run
python scripts/run_mqtt_test.py --mode subscriber --endpoint xxxxxx-ats.iot.us-east-1.amazonaws.com --port 8883 --tls --cert-dir ./certs --topic dji/matrice4td/inference
```

**Code example for your backend:**

```python
from src.mqtt.aws_subscriber import DJIInferenceSubscriber

def my_callback(inference):
    # Custom logic
    if inference.has_person_without_hardhat():
        print(f"ALERT! {len(inference.get_safety_violations())} violations")
        # Send to Slack, save to DB, etc.

subscriber = DJIInferenceSubscriber(
    endpoint="xxxxxx-ats.iot.us-east-1.amazonaws.com",
    topic="dji/matrice4td/inference",
    use_tls=True,
    cert_dir="./certs",
    on_detection_callback=my_callback
)
subscriber.start()
```

### 2.4 AWS IoT Rule (Optional - Forward to S3/Lambda)

1. IoT Core → Message routing → Rules → Create
2. Name: `dji_to_s3`
3. SQL: `SELECT * FROM 'dji/matrice4td/inference'`
4. Actions:
   - S3: Bucket `dji-inference-logs`, Key `${timestamp()}.json`
   - Lambda: Function `process_safety_violation`
   - Republish: Topic `alerts/safety` if No-Hard-Hat detected

Rule SQL for alerts only:
```sql
SELECT * FROM 'dji/matrice4td/inference' WHERE contains(detections[*].class_name, 'No-Hard-Hat')
```

## 2.5 Validation for Deliverable

To prove successful validation that team can receive MQTT JSON in AWS broker:

1. Run subscriber for 5 minutes
2. Fly drone or use mock publisher
3. Collect `demo/dji_inference_log.jsonl` - should have ~900 entries (3fps * 300s)
4. Show AWS IoT MQTT test console screenshot with messages arriving
5. Show safety alerts log

Include these in SOP deliverable.

## Payload Format

See `demo/sample_payload.json` for full example.

Simplified:
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

## Security Notes

- Never commit certs to git (in .gitignore)
- Use least privilege policy (restrict to dji/* topics)
- Rotate certs every 90 days
- Enable CloudWatch logs for IoT Core
