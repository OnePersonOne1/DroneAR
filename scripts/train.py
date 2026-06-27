#!/usr/bin/env python3
"""Train a YOLO26 drone detector for Magic Leap 2 deployment.

ML2-driven choices (do not change without reason):
  - yolo26n (nano) is the primary on-device model; yolo26s is the accuracy comparison.
  - NMS-free one-to-one head is kept (default) so the device needs no NMS.
  - imgsz=640 baseline; imgsz=960 / P2 head are the small-object levers (dataset is
    ~77% small objects).

Environment fixes baked into the workflow (see README "Troubleshooting"):
  - polars 1.42 SIGBUS on import on this CPU -> use polars-lts-cpu (ultralytics reads
    results.csv via polars after every epoch, so a broken polars kills training at the
    first checkpoint save). requirements.txt pins polars-lts-cpu.
  - cache="disk" is the default (spec preference); disk space has been freed. cache="ram"
    can SIGBUS via DataLoader shared-memory; override with --cache disk|ram|False.
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    ap = argparse.ArgumentParser(description="YOLO26 drone training")
    ap.add_argument("--model", default="yolo26n.pt")
    ap.add_argument("--data", default="configs/dut_drone.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--batch", type=float, default=-1, help="-1 = auto")
    ap.add_argument("--device", default="0")
    ap.add_argument("--cache", default="disk", help='disk | ram | False')
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--project", default="runs")
    ap.add_argument("--name", default="yolo26n_drone_640")
    return ap.parse_args()


def coerce_cache(v):
    if str(v).lower() in ("false", "0", "none", ""):
        return False
    return str(v)


def main():
    a = parse_args()
    batch = int(a.batch) if float(a.batch).is_integer() else a.batch
    # Absolute project dir so output is always <repo>/runs/<name> (ultralytics otherwise
    # nests it under SETTINGS.runs_dir when the project path is relative).
    project = str(Path(a.project).resolve())
    model = YOLO(a.model)
    results = model.train(
        data=a.data,
        imgsz=a.imgsz,
        epochs=a.epochs,
        patience=a.patience,
        batch=batch,
        device=a.device,
        cache=coerce_cache(a.cache),
        workers=a.workers,
        project=project,
        name=a.name,
        plots=True,
    )
    # Best checkpoint location for downstream phases.
    print(f"\nBEST_WEIGHTS: {model.trainer.best}")
    try:
        print(f"FINAL_mAP50: {results.box.map50:.4f}  mAP50-95: {results.box.map:.4f}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
