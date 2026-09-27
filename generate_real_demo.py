"""
Real inference demo generator - uses YOUR trained best.pt
Produces 100% real YOLO detections, no fake boxes on walls

Usage:
  python generate_real_demo.py --model runs/detect/train/weights/best.pt --source data/yolo_visdrone/images/val --output demo/real --num 5
  python generate_real_demo.py --model D:\dji-matrice-4td-yolo-pipeline\runs\detect\train\weights\best.pt --source datasets/images/val --output demo/real --num 5
"""
import argparse
from pathlib import Path
from ultralytics import YOLO
import cv2
import random

def main():
    parser = argparse.ArgumentParser(description="Generate REAL detection images from your trained model")
    parser.add_argument("--model", required=True, help="Path to best.pt")
    parser.add_argument("--source", required=True, help="Path to val images folder")
    parser.add_argument("--output", default="demo/real", help="Output folder")
    parser.add_argument("--num", type=int, default=5, help="Number of images to process")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()

    model_path = Path(args.model)
    source_path = Path(args.source)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print(f"   Check: runs/detect/train/weights/best.pt or runs/detect/train2/weights/best.pt")
        return
    if not source_path.exists():
        print(f"❌ Source not found: {source_path}")
        print(f"   Try: data/yolo_visdrone/images/val, datasets/images/val, data/final_4class/images/val")
        # try to find val folders
        for p in Path(".").rglob("images/val"):
            print(f"   Found candidate: {p}")
        return

    print(f"✅ Loading model: {model_path}")
    model = YOLO(str(model_path))
    
    # Get image list
    images = list(source_path.glob("*.jpg")) + list(source_path.glob("*.png")) + list(source_path.glob("*.jpeg"))
    if not images:
        print(f"❌ No images in {source_path}")
        return
    
    print(f"✅ Found {len(images)} val images, picking {args.num} random")
    selected = random.sample(images, min(args.num, len(images)))

    for i, img_path in enumerate(selected):
        print(f"\n[{i+1}/{len(selected)}] {img_path.name}")
        results = model.predict(str(img_path), conf=args.conf, imgsz=640, save=False, verbose=False)
        
        # Plot with boxes (real YOLO output)
        for result in results:
            # result.plot() returns annotated image (BGR)
            annotated = result.plot()  # numpy array
            # Save
            out_file = output_path / f"real_{i+1:02d}_{img_path.stem}_conf{args.conf}.jpg"
            cv2.imwrite(str(out_file), annotated)
            # Print detections
            boxes = result.boxes
            if boxes is not None:
                print(f"   Detections: {len(boxes)}")
                for box in boxes:
                    cls_id = int(box.cls[0])
                    cls_name = model.names[cls_id]
                    conf = float(box.conf[0])
                    print(f"     - {cls_name} {conf:.2f}")
            else:
                print(f"   No detections at conf {args.conf}")

    print(f"\n✅ Done! Real images saved to: {output_path.absolute()}")
    print(f"   Now copy to demo/: copy {output_path}/* demo/")

if __name__ == "__main__":
    main()
