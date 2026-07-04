#!/usr/bin/env python3
"""원거리(소형) 드론 미탐(FN) 정량화 — GT 크기 bin별 recall/FN 히스토그램.

문제 정의: "먼 거리 드론이 잘 탐지되지 않는다" → 몇 px부터 recall이 무너지는지 실측.
far-drone 지표 정의: 정규화변 sqrt(w·h)×640 < FAR_PX (기본 16px) 인 GT 의 recall.

모델(old/100ep/300ep) × test set(DUT-test/Maciullo-test) 전 조합을 돌려
reports/fn_size_analysis.{md,json} 에 bin별 recall 매트릭스를 쓴다.

매칭 규칙은 scripts/eval_compare.py 와 동일(conf 0.25, greedy IoU>=0.5).
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

BINS = [(0, 8), (8, 16), (16, 24), (24, 32), (32, 64), (64, 128), (128, 10 ** 9)]
BIN_LABELS = ["<8", "8-16", "16-24", "24-32", "32-64", "64-128", "128+"]


def iou_xyxy(a, b):
    xa, ya = max(a[0], b[0]), max(a[1], b[1])
    xb, yb = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    ua = max(0.0, a[2]-a[0])*max(0.0, a[3]-a[1]) + max(0.0, b[2]-b[0])*max(0.0, b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def load_gt(label_path, W, H):
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


def bin_of(side):
    for i, (lo, hi) in enumerate(BINS):
        if lo <= side < hi:
            return i
    return len(BINS) - 1


def run(model, img_dir, lbl_dir, imgsz, device, conf, iou_match):
    matched = np.zeros(len(BINS), dtype=int)
    total = np.zeros(len(BINS), dtype=int)
    torch.cuda.empty_cache()
    for res in model.predict(source=str(img_dir), imgsz=imgsz, device=device,
                             conf=conf, iou=0.7, verbose=False, stream=True, batch=16):
        p = Path(res.path)
        H, W = res.orig_shape
        gts = load_gt(lbl_dir / f"{p.stem}.txt", W, H)
        dets = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.empty((0, 4))
        used = [False] * len(gts)
        for d in dets:
            best, bj = iou_match, -1
            for gj, g in enumerate(gts):
                if used[gj]:
                    continue
                v = iou_xyxy(d, g[:4])
                if v >= best:
                    best, bj = v, gj
            if bj >= 0:
                used[bj] = True
        for gj, g in enumerate(gts):
            b = bin_of(g[4])
            total[b] += 1
            if used[gj]:
                matched[b] += 1
    return matched, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", default="/mnt/ssd_0/dataset/merged_drone")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou-match", type=float, default=0.5)
    ap.add_argument("--far-px", type=float, default=16.0)
    ap.add_argument("--out", default="reports/fn_size_analysis")
    ap.add_argument("--models", nargs="+", default=[
        "old_DUT_only:weights/yolo26n_drone_640.pt",
        "merged_100ep:weights/yolo26n_drone_640_mergedataset_100epoch.pt",
        "merged_300ep:weights/yolo26n_drone_640_mergedataset_300epoch.pt"])
    a = ap.parse_args()
    merged = Path(a.merged)
    sets = {"DUT-test": ("test_dut",), "Maciullo-test": ("test_maciullo",)}

    out = {}
    for spec in a.models:
        name, path = spec.split(":", 1)
        if not Path(path).exists():
            print(f"[skip] {name}: {path}")
            continue
        model = YOLO(path)
        out[name] = {}
        for sname, (sub,) in sets.items():
            m, t = run(model, merged/"images"/sub, merged/"labels"/sub,
                       a.imgsz, a.device, a.conf, a.iou_match)
            rec = [round(mi/ti, 4) if ti else None for mi, ti in zip(m, t)]
            far_idx = [i for i, (lo, hi) in enumerate(BINS) if hi <= a.far_px]
            far_m, far_t = int(m[far_idx].sum()), int(t[far_idx].sum())
            out[name][sname] = {
                "bins_px640": BIN_LABELS,
                "gt_total": [int(x) for x in t],
                "gt_matched": [int(x) for x in m],
                "recall_by_bin": rec,
                f"far_recall(<{a.far_px:g}px)": round(far_m/far_t, 4) if far_t else None,
                "far_gt": far_t, "far_fn": far_t - far_m,
            }
            print(f"{name:14} {sname:14} far(<{a.far_px:g}px) recall="
                  f"{out[name][sname][f'far_recall(<{a.far_px:g}px)']} "
                  f"({far_m}/{far_t})  bins={rec}")

    Path(a.out + ".json").write_text(json.dumps(out, indent=2))
    L = ["# FN 크기 분석 — 원거리(소형) 드론 미탐 정량화", "",
         f"conf={a.conf}, IoU매칭 0.5, imgsz={a.imgsz}. 크기 = 정규화변 sqrt(w·h)×640 (px).",
         f"**far-drone 지표 = <{a.far_px:g}px GT recall.**", ""]
    for name in out:
        for sname in out[name]:
            d = out[name][sname]
            L += [f"## {name} — {sname}", "",
                  "| bin(px@640) | " + " | ".join(BIN_LABELS) + " | far(<%g) |" % a.far_px,
                  "|---|" + "---:|" * (len(BIN_LABELS) + 1)]
            L.append("| GT 수 | " + " | ".join(str(x) for x in d["gt_total"]) +
                     f" | {d['far_gt']} |")
            L.append("| recall | " + " | ".join(str(x) for x in d["recall_by_bin"]) +
                     f" | **{d[f'far_recall(<{a.far_px:g}px)']}** |")
            L.append("")
    Path(a.out + ".md").write_text("\n".join(L) + "\n")
    print(f"\nsaved {a.out}.md / .json")


if __name__ == "__main__":
    main()
