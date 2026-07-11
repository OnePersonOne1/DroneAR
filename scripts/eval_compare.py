#!/usr/bin/env python3
"""Phase 4 — evaluate & compare DUT-only (old) vs merged (new) on fixed test sets.

Test sets (held out, untouched originals):
    DUT-test        : merged_drone/images/test_dut       (+ labels/test_dut)
    Maciullo-test   : merged_drone/images/test_maciullo  (+ labels/test_maciullo)

For each (model, test set):
  A) ultralytics val()  -> mAP@0.5, mAP@0.5:0.95, precision, recall     (authoritative)
  B) failure-mode pass (conf>=CONF, greedy IoU match @0.5):
       - background FP rate : detections not matching any GT, per image
                              (+ FP on GT-empty images if any = pure background clutter)
       - small-object recall: recall for GT with side@640 < SMALL_PX
                              (ground-background misses proxy), plus med / large

Writes reports/old_vs_new.md (+ .json). No ONNX export (deferred by spec).
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO


def iou_xyxy(a, b):
    xa, ya = max(a[0], b[0]), max(a[1], b[1])
    xb, yb = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    ua = max(0.0, a[2]-a[0])*max(0.0, a[3]-a[1]) + max(0.0, b[2]-b[0])*max(0.0, b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def load_gt(label_path, W, H):
    """YOLO norm -> list of xyxy(px) at (W,H) + side@640 for size bucketing."""
    gts = []
    if not label_path.exists():
        return gts
    for ln in label_path.read_text().splitlines():
        p = ln.split()
        if len(p) != 5:
            continue
        cx, cy, w, h = (float(v) for v in p[1:])
        x1, y1 = (cx - w/2)*W, (cy - h/2)*H
        x2, y2 = (cx + w/2)*W, (cy + h/2)*H
        side640 = math.sqrt(max(w*h, 0.0)) * 640
        gts.append([x1, y1, x2, y2, side640])
    return gts


def ul_metrics(model, data_yaml, imgsz, device):
    r = model.val(data=data_yaml, split="val", imgsz=imgsz, device=device,
                  verbose=False, plots=False, project="runs",
                  name=f"cmp_{Path(data_yaml).stem}", exist_ok=True)
    b = r.box
    return {"mAP50": round(float(b.map50), 4), "mAP50_95": round(float(b.map), 4),
            "precision": round(float(b.mp), 4), "recall": round(float(b.mr), 4)}


def failure_modes(model, img_dir, lbl_dir, imgsz, device, conf, iou_match, small_px, large_px):
    imgs = sorted(img_dir.glob("*.jpg"))
    n_img = len(imgs)
    tp = fp = 0
    fp_on_empty = n_empty = 0
    # size-bucketed GT recall
    buck = {"small": [0, 0], "medium": [0, 0], "large": [0, 0]}  # [matched, total]
    torch.cuda.empty_cache()
    for res in model.predict(source=str(img_dir), imgsz=imgsz, device=device,
                             conf=conf, iou=0.7, verbose=False, stream=True, batch=16):
        p = Path(res.path)
        H, W = res.orig_shape
        gts = load_gt(lbl_dir / f"{p.stem}.txt", W, H)
        dets = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.empty((0, 4))
        used = [False] * len(gts)
        # match each detection to best unused GT
        matched_det = [False] * len(dets)
        for di, d in enumerate(dets):
            best, bj = iou_match, -1
            for gj, g in enumerate(gts):
                if used[gj]:
                    continue
                v = iou_xyxy(d, g[:4])
                if v >= best:
                    best, bj = v, gj
            if bj >= 0:
                used[bj] = True
                matched_det[di] = True
        tp += sum(matched_det)
        fp_i = sum(1 for m in matched_det if not m)
        fp += fp_i
        if len(gts) == 0:
            n_empty += 1
            fp_on_empty += len(dets)
        # size buckets on GT
        for gj, g in enumerate(gts):
            s = g[4]
            b = "small" if s < small_px else ("large" if s > large_px else "medium")
            buck[b][1] += 1
            if used[gj]:
                buck[b][0] += 1
    def rec(x):
        return round(x[0]/x[1], 4) if x[1] else None
    return {
        "images": n_img,
        "TP": tp, "FP": fp,
        "FP_per_image": round(fp / max(n_img, 1), 4),
        "GT_empty_images": n_empty, "FP_on_empty_images": fp_on_empty,
        "recall_small_<{}px".format(small_px): rec(buck["small"]), "n_small": buck["small"][1],
        "recall_medium": rec(buck["medium"]), "n_medium": buck["medium"][1],
        "recall_large_>{}px".format(large_px): rec(buck["large"]), "n_large": buck["large"][1],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default="weights/yolo26/yolo26n_drone_640.pt")
    ap.add_argument("--new", default="runs/merged_yolo26n/weights/best.pt")
    ap.add_argument("--merged", default="/mnt/ssd_0/dataset/merged_drone")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou-match", type=float, default=0.5)
    ap.add_argument("--small-px", type=float, default=32.0)
    ap.add_argument("--large-px", type=float, default=96.0)
    a = ap.parse_args()
    merged = Path(a.merged)

    sets = {
        "DUT-test": ("configs/eval_test_dut.yaml", merged/"images"/"test_dut", merged/"labels"/"test_dut"),
        "Maciullo-test": ("configs/eval_test_maciullo.yaml", merged/"images"/"test_maciullo", merged/"labels"/"test_maciullo"),
    }
    models = {"old_DUT_only": a.old, "new_merged": a.new}

    out = {}
    for mname, mpath in models.items():
        if not Path(mpath).exists():
            print(f"[skip] {mname}: {mpath} not found"); continue
        model = YOLO(mpath)
        out[mname] = {}
        for sname, (yaml, idir, ldir) in sets.items():
            m = ul_metrics(model, yaml, a.imgsz, a.device)
            fm = failure_modes(model, idir, ldir, a.imgsz, a.device, a.conf,
                               a.iou_match, a.small_px, a.large_px)
            out[mname][sname] = {"metrics": m, "failure_modes": fm}
            print(f"{mname:14} {sname:14} mAP50={m['mAP50']:.4f} mAP50-95={m['mAP50_95']:.4f} "
                  f"P={m['precision']:.4f} R={m['recall']:.4f} | FP/img={fm['FP_per_image']} "
                  f"small_recall={fm[[k for k in fm if k.startswith('recall_small')][0]]}")

    Path("reports").mkdir(exist_ok=True)
    Path("reports/old_vs_new.json").write_text(json.dumps(out, indent=2))

    # markdown
    disp = {"old_DUT_only": "old (DUT-only)", "new_merged": "new (merged)"}
    def row(mn, sn):
        m = out[mn][sn]["metrics"]; f = out[mn][sn]["failure_modes"]
        sk = [k for k in f if k.startswith("recall_small")][0]
        return (f"| {disp.get(mn, mn)} | {sn} | {m['mAP50']} | {m['mAP50_95']} | {m['precision']} | "
                f"{m['recall']} | {f['FP_per_image']} | {f[sk]} |")
    L = ["# old vs new 평가 (생성물)", "",
         "> `scripts/eval_compare.py` 산출물 — 재실행 시 덮어써짐. 정확도 비교의 단일 출처는 "
         "[ablation_matrix.md](ablation_matrix.md).", "",
         "## 결과 (고정 held-out test)", "",
         "| 모델 | test set | mAP@0.5 | mAP@0.5:0.95 | P | R | FP/img | small-recall(<32px) |",
         "|---|---|---:|---:|---:|---:|---:|---:|"]
    for sn in sets:
        for mn in out:
            L.append(row(mn, sn))
    L += ["", "> 비교 조건: best.pt vs best.pt, 동일 test set. old 150ep(best@134) / new 100ep "
          "— 이미지 노출량은 new(5.66M)가 old(0.70M)보다 많아 epoch cap이 new를 불리하게 하지 않음.",
          "", "## Failure-mode detail", "```",
          json.dumps({mn: {sn: out[mn][sn]["failure_modes"] for sn in out[mn]} for mn in out}, indent=2),
          "```"]
    Path("reports/old_vs_new.md").write_text("\n".join(L) + "\n")
    print("\nsaved reports/old_vs_new.md")


if __name__ == "__main__":
    main()
