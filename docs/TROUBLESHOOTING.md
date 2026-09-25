# Troubleshooting Guide - DJI Matrice 4TD Deployment

## Training Issues

### CUDA out of memory
- Reduce batch from 16 to 8 or 4
- Use `imgsz=640` not larger
- Close other GPU processes

### Low mAP (<70%)
- Check dataset validation: `python scripts/download_dataset.py --validate`
- Class imbalance - ensure all 4 classes have >500 images
- Increase epochs to 150
- Check label format - YOLO format must be normalized 0-1

## ONNX Export Issues

### `SiLU not supported` or `Unsupported op`
- DJI NPU may not support SiLU activation
- Fix: Try opset 11 instead of 12, or use custom model with ReLU
- In Ultralytics, SiLU is default - need to modify model.yaml to use ReLU (advanced)

### Input shape not [1,3,640,640]
- Ensure `dynamic=False` in export
- Check `imgsz=640` not 640,640 tuple
- Validate with `src/export/validator.py`

### ONNX checker fails
- Update onnx: `pip install -U onnx`
- Try `simplify=False`
- Check PyTorch version compatibility

## DJI Portal Quantization Issues

### Quantization fails immediately
- Check zip structure - must have onnx at root, not nested
- Ensure calibration_images has 100-200 images, not 0
- Check classes.txt has 4 lines matching yaml

### Accuracy drop >5% after INT8
- Calibration set not diverse - regenerate with `strategy=diverse`
- Include all lighting conditions, angles, distances
- Ensure calibration images are from training set, not val/test

### Model too large after quant (still >10MB)
- You used yolov8s/m/l instead of n - retrain with yolov8n.pt
- Check if you exported with `quantize=8` locally - DJI expects FP32 input

### Portal says "Model bound to another SN"
- Model already bound - need to unbind in Device Management or create new version
- Each .dji package bound to one SN for licensing

## Pilot 2 Deployment Issues

### Model not showing in Pilot 2 after sync
- Check RC Plus 2 internet connection
- Check Pilot 2 version - update to latest
- Check DJI account logged in same as portal account
- Check SN binding in portal - must be same drone

### Bounding boxes not showing in live view
- Check model enabled in AI settings
- Check confidence threshold not too high (set 0.3 for testing)
- Check camera is main wide camera, not zoom (some models only work on wide)
- Restart drone and RC

### Inference very slow (<5fps)
- Model too heavy - ensure yolov8n
- Check if other AI features enabled - disable
- NPU may be thermal throttling - let drone cool

## MQTT Issues

### Subscriber not receiving messages
- Check broker running: `docker ps | grep emqx`
- Check topic matches publisher and subscriber (exact string)
- Check firewall - port 1883 open
- For AWS: Check certs path, endpoint, policy allows Publish/Subscribe

### Mock publisher fails to connect
- EMQX not running - `docker-compose -f docker/docker-compose.yml up -d`
- Check port 1883 not used by other service
- Try `mosquitto_pub` to test broker: `mosquitto_pub -h localhost -t test -m hello`

### FPS not ~3, but higher/lower
- Mock publisher fps param controls publish rate, not inference rate
- Real drone: FPS set in FlightHub 2 AI settings (3fps)
- If subscriber shows higher FPS, multiple drones publishing to same topic - use specific topic per drone

### TLS error for AWS
- Certs not found - check `./certs/` has 3 files
- Endpoint wrong - must be ATS endpoint (contains -ats-)
- Policy doesn't allow Connect - add `iot:Connect` to policy
- Time skew - ensure system time correct (TLS fails if time wrong)

## AWS IoT Issues

### Certificate rejected
- Policy not attached to cert
- Cert not attached to Thing
- Endpoint region mismatch (cert for us-east-1 but endpoint us-west-2)

### Rule not triggering
- SQL syntax - test in IoT Core → Test
- Topic doesn't match Rule SQL FROM clause
- Role for S3/Lambda doesn't have permissions

## General

### Colab out of memory
- Reduce batch to 8
- Use `device=cpu` for export only
- Restart runtime and run only training cell

### Import errors (src not found)
- Ensure `PYTHONPATH` includes project root: `export PYTHONPATH=$PWD`
- Or `sys.path.insert(0, ...)` in notebook first cell

### Log files not created
- Check `demo/` dir exists - `mkdir -p demo`
- Check write permissions

## Getting Help

1. Check DJI Developer Forum: https://developer.dji.com/forum/
2. Check RIIS tutorial: https://www.riis.com/blog/deploying-custom-ml-models-on-the-dji-matrice-4e
3. Check Reddit r/dji: https://www.reddit.com/r/dji/
4. For this repo: Open issue with logs from `runs/` and `dji_submission/`
