# Architecture - DJI Matrice 4TD YOLO Pipeline

## System Overview

This pipeline is designed for production deployment to edge NPU with mentorship focus.

### Components

```
┌─────────────────────────────────────────────────────────────────┐
│                        Dataset Layer                            │
│  - Roboflow Hard-Hat Workers (2k images)                        │
│  - COCO Person (3k) + Vehicle (2k)                              │
│  - Validator: YOLO format, class balance                        │
│  - Calibration Generator: 150 diverse images for INT8           │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│                      Training Layer                             │
│  - Model: YOLOv8n (3.2M params, 8.7 GFLOPs)                     │
│  - Input: 640x640 static (DJI standard)                         │
│  - Augmentations: No rotation, perspective warp (drone oblique) │
│  - Optimizer: AdamW, Early stopping                             │
│  - Output: best.pt + metrics                                    │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│                       Export Layer                              │
│  - ONNX Exporter: opset 12, static, simplified                  │
│  - Validator: onnx.checker, input shape [1,3,640,640]          │
│  - Benchmark: CPU latency ~80ms, NPU ~45ms after INT8          │
│  - Fallback: MMYOLO exporter for .pth if DJI requires          │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│                    DJI Deployment Layer                         │
│  - Package Generator: zip with onnx + classes.txt + calib      │
│  - Portal: ai-developer.dji.com → Quantization INT8 → .dji     │
│  - Binding: SN-based licensing                                  │
│  - Pilot 2: Sync, enable, live bbox overlay                     │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│                      MQTT Layer                                 │
│  - Mock Publisher: Simulates Matrice 4TD @ 3fps JSON            │
│  - Subscriber: paho-mqtt, TLS for AWS IoT Core                  │
│  - Payload Models: Pydantic validation, safety alerts           │
│  - AWS IoT: Rule to S3/Lambda, topic dji/matrice4td/inference  │
│  - Log: JSONL for deliverable validation (900 msgs in 5min)     │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Training Data:** 4 classes, YOLO format, 640x640
2. **Model:** YOLOv8n trained with drone-specific aug
3. **ONNX:** FP32 6-12MB, validated
4. **Calibration:** 150 images covering all classes
5. **DJI Portal:** INT8 PTQ → 3-5MB .dji package
6. **Deployment:** Bound to SN, deployed via Pilot 2
7. **Inference:** NPU @ 22fps, throttled to 3fps JSON
8. **MQTT:** JSON payload to AWS IoT Core
9. **Backend:** Subscriber validates, alerts on No-Hard-Hat

### Why YOLOv8n?

| Model | Params | GFLOPs | Size FP32 | Size INT8 | NPU Latency | Fits 4TD? |
|-------|--------|--------|-----------|-----------|-------------|-----------|
| YOLOv8n | 3.2M | 8.7 | 6MB | 3MB | 45ms | ✅ Yes |
| YOLOv8s | 11M | 28.6 | 22MB | 11MB | 120ms | ⚠️ Marginal |
| YOLOv8m | 25M | 78.9 | 52MB | 26MB | 300ms | ❌ No |

Matrice 4TD NPU estimated 10-15 TOPS, limited memory - only n is viable.

### Why ONNX opset 12?

- DJI docs recommend 11-14
- 12 is most compatible with NPU toolchain
- Static shape required - dynamic causes quantization failure
- Simplified graph removes training-only ops

### Why JSON @ 3fps not video?

- Bandwidth: Video 5-10 Mbps vs JSON 5-10 KBps (1000x savings)
- Privacy: No video stored, only metadata
- Battery: Less TX power
- Cost: Less S3 storage

### Security

- Certs never committed (gitignore)
- TLS 1.2 for AWS IoT
- Least privilege policy (dji/* topics only)
- SN binding prevents model theft

### Scalability

- Training: Can scale to 1000s images with same pipeline
- Deployment: Same pipeline works for any new classes (just update yaml)
- MQTT: One topic per drone, can handle 100s drones with same subscriber pattern
- AWS: IoT Rule can fan-out to multiple consumers (S3, Lambda, Kinesis)

### Production Extras

Beyond basic requirements, this pipeline includes:

- Pydantic models: Type safety, catches malformed JSON early
- Mock publisher: Test without hardware, CI/CD friendly
- Dataset validator: Prevents garbage-in-garbage-out
- Benchmark: Proves latency before deployment
- Docker: Reproducible env, no "works on my machine"
- Alert logic: Business value (safety violation) not just detection
- Comprehensive error handling and logging
