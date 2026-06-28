#!/usr/bin/env python3
"""Evaluate trained YOLO26 drone weights on the val and test splits; write metrics.json.

Usage:
    python scripts/eval.py --weights weights/yolo26n_drone_640.pt weights/yolo26s_drone_640.pt
"""
import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="+", default=["weights/yolo26n_drone_640.pt"])
    ap.add_argument("--data", default="configs/dut_drone.yaml")
    ap.add_argument("--imgsz", type=int, default=0,
                    help="0 = 모델명 끝 숫자에서 자동(yolo26n_drone_960 -> 960)")
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", default="weights/metrics.json")
    return ap.parse_args()


def eval_split(weights, data, imgsz, device, split):
    m = YOLO(weights)
    r = m.val(data=data, split=split, imgsz=imgsz, device=device, verbose=False,
              plots=False, project="runs", name=f"val_{Path(weights).stem}_{split}",
              exist_ok=True)
    b = r.box
    return {"mAP50": round(float(b.map50), 4), "mAP50_95": round(float(b.map), 4),
            "precision": round(float(b.mp), 4), "recall": round(float(b.mr), 4)}


def resolve_imgsz(weights, override):
    if override:
        return override
    import re
    m = re.search(r"(\d+)$", Path(weights).stem)   # ..._960 -> 960
    return int(m.group(1)) if m else 640


def main():
    a = parse_args()
    out = {}
    for w in a.weights:
        stem = Path(w).stem
        imgsz = resolve_imgsz(w, a.imgsz)
        out[stem] = {"imgsz": imgsz}
        for split in ("val", "test"):
            out[stem][split] = eval_split(w, a.data, imgsz, a.device, split)
            s = out[stem][split]
            print(f"{stem:24} {split:4}  mAP50={s['mAP50']:.4f}  "
                  f"mAP50-95={s['mAP50_95']:.4f}  P={s['precision']:.4f}  R={s['recall']:.4f}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
