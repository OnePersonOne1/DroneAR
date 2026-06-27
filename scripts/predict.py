#!/usr/bin/env python3
"""Run a trained YOLO26 drone detector on an image or folder and save annotated outputs.

Examples:
    python scripts/predict.py --weights weights/yolo26n_drone_640.pt --source some/dir
    python scripts/predict.py --weights runs/yolo26n_drone_640/weights/best.pt \
        --source /mnt/ssd_0/dataset/dut_yolo/images/test --max 12
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/yolo26n_drone_640.pt")
    ap.add_argument("--source", default="/mnt/ssd_0/dataset/dut_yolo/images/test")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="0")
    ap.add_argument("--max", type=int, default=12, help="cap number of images from a folder")
    ap.add_argument("--out", default="runs/predict_demo")
    return ap.parse_args()


def main():
    a = parse_args()
    src = Path(a.source)
    if src.is_dir():
        imgs = sorted(p for p in src.iterdir()
                      if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"))[:a.max]
        source = [str(p) for p in imgs]
    else:
        source = str(src)

    out = Path(a.out).resolve()                       # absolute -> avoids ultralytics nesting
    model = YOLO(a.weights)
    results = model.predict(source=source, imgsz=a.imgsz, conf=a.conf, device=a.device,
                            save=True, project=str(out.parent),
                            name=out.name, exist_ok=True)
    n_det = sum(len(r.boxes) for r in results)
    print(f"images={len(results)}  detections={n_det}  saved -> {a.out}")


if __name__ == "__main__":
    main()
