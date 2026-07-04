#!/usr/bin/env python3
"""SAHI 스타일 타일 추론 벤치 — 재학습 없이 원거리(소형) recall 을 얼마나 회복하는가.

고정 모델(기본 merged-300ep)로 DUT-test(네이티브 1280×720)에서 3개 추론 모드 비교:
  base640  : letterbox 640 단일 패스 (현행 배포 방식)
  hi1280   : letterbox 1280 단일 패스 (다운스케일 없음 — 클라우드 4090 여유)
  tile640  : 640×640 타일(overlap 0.2) + 전체 프레임 패스 → 전역 좌표 병합 + NMS

지표: 크기 bin별 recall(analyze_fn.py 와 동일 bin), far(<16px) recall, FP/img, ms/img.
Maciullo-test 는 네이티브 640×480 이라 타일링이 무의미(단일 타일) → DUT-test 만 벤치.
산출: reports/sahi_bench.{md,json}
"""
import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

BINS = [(0, 8), (8, 16), (16, 24), (24, 32), (32, 64), (64, 128), (10 ** 9,) * 0 or (128, 10 ** 9)]
BIN_LABELS = ["<8", "8-16", "16-24", "24-32", "32-64", "64-128", "128+"]


def iou_xyxy(a, b):
    xa, ya = max(a[0], b[0]), max(a[1], b[1])
    xb, yb = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    ua = max(0.0, a[2]-a[0])*max(0.0, a[3]-a[1]) + max(0.0, b[2]-b[0])*max(0.0, b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def nms(boxes, iou_thr=0.5):
    """boxes (N,5)=[x1,y1,x2,y2,score] class-agnostic."""
    if len(boxes) == 0:
        return boxes
    boxes = boxes[boxes[:, 4].argsort()[::-1]]
    keep = []
    while len(boxes):
        keep.append(boxes[0])
        if len(boxes) == 1:
            break
        rest = boxes[1:]
        ious = np.array([iou_xyxy(keep[-1], b) for b in rest])
        boxes = rest[ious < iou_thr]
    return np.stack(keep)


def load_gt(label_path, W, H):
    gts = []
    if not label_path.exists():
        return gts
    for ln in label_path.read_text().splitlines():
        p = ln.split()
        if len(p) != 5:
            continue
        cx, cy, w, h = (float(v) for v in p[1:])
        gts.append([(cx-w/2)*W, (cy-h/2)*H, (cx+w/2)*W, (cy+h/2)*H,
                    math.sqrt(max(w*h, 0.0)) * 640])
    return gts


def bin_of(side):
    for i, (lo, hi) in enumerate(BINS):
        if lo <= side < hi:
            return i
    return len(BINS) - 1


def tile_origins(W, H, tile, overlap):
    step = int(tile * (1 - overlap))
    xs = list(range(0, max(W - tile, 0) + 1, step))
    if xs[-1] != W - tile and W > tile:
        xs.append(W - tile)
    ys = list(range(0, max(H - tile, 0) + 1, step))
    if ys[-1] != H - tile and H > tile:
        ys.append(H - tile)
    return [(x, y) for y in ys for x in xs]


def det_single(model, img, imgsz, conf, device):
    r = model.predict(img, imgsz=imgsz, conf=conf, device=device,
                      iou=0.7, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return np.empty((0, 5))
    b = r.boxes.xyxy.cpu().numpy()
    s = r.boxes.conf.cpu().numpy()[:, None]
    return np.concatenate([b, s], 1)


def det_tiled(model, img, tile, overlap, conf, device):
    H, W = img.shape[:2]
    crops, origins = [], []
    for (x, y) in tile_origins(W, H, tile, overlap):
        crops.append(np.ascontiguousarray(img[y:y+tile, x:x+tile]))
        origins.append((x, y))
    crops.append(img)                    # 전체 프레임 패스 (대형 객체 안전망)
    origins.append(None)
    rs = model.predict(crops, imgsz=tile, conf=conf, device=device,
                       iou=0.7, verbose=False, batch=len(crops))
    out = []
    for r, org in zip(rs, origins):
        if r.boxes is None or len(r.boxes) == 0:
            continue
        b = r.boxes.xyxy.cpu().numpy()
        s = r.boxes.conf.cpu().numpy()[:, None]
        if org is not None:
            b[:, [0, 2]] += org[0]
            b[:, [1, 3]] += org[1]
        out.append(np.concatenate([b, s], 1))
    if not out:
        return np.empty((0, 5))
    return nms(np.concatenate(out), 0.5)


def evaluate(mode, model, imgs, lbl_dir, a):
    matched = np.zeros(len(BINS), dtype=int)
    total = np.zeros(len(BINS), dtype=int)
    fp = 0
    times = []
    for p in imgs:
        img = cv2.imread(str(p))
        H, W = img.shape[:2]
        t0 = time.perf_counter()
        if mode == "base640":
            dets = det_single(model, img, 640, a.conf, a.device)
        elif mode == "hi1280":
            dets = det_single(model, img, 1280, a.conf, a.device)
        else:  # tile640
            dets = det_tiled(model, img, 640, a.overlap, a.conf, a.device)
        times.append((time.perf_counter() - t0) * 1000)
        gts = load_gt(lbl_dir / f"{p.stem}.txt", W, H)
        used = [False] * len(gts)
        n_match = 0
        for d in dets:
            best, bj = a.iou_match, -1
            for gj, g in enumerate(gts):
                if used[gj]:
                    continue
                v = iou_xyxy(d[:4], g[:4])
                if v >= best:
                    best, bj = v, gj
            if bj >= 0:
                used[bj] = True
                n_match += 1
        fp += len(dets) - n_match
        for gj, g in enumerate(gts):
            b = bin_of(g[4])
            total[b] += 1
            if used[gj]:
                matched[b] += 1
    rec = [round(m/t, 4) if t else None for m, t in zip(matched, total)]
    far_idx = [i for i, (lo, hi) in enumerate(BINS) if hi <= a.far_px]
    fm, ft = int(matched[far_idx].sum()), int(total[far_idx].sum())
    return {"recall_by_bin": rec, "gt_total": [int(x) for x in total],
            "far_recall": round(fm/ft, 4) if ft else None, "far_gt": ft,
            "overall_recall": round(float(matched.sum()/total.sum()), 4),
            "FP_per_image": round(fp/len(imgs), 4),
            "ms_per_image": round(float(np.mean(times)), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/yolo26n_drone_640_mergedataset_300epoch.pt")
    ap.add_argument("--img-dir", default="/mnt/ssd_0/dataset/merged_drone/images/test_dut")
    ap.add_argument("--lbl-dir", default="/mnt/ssd_0/dataset/merged_drone/labels/test_dut")
    ap.add_argument("--device", default="0")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou-match", type=float, default=0.5)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--far-px", type=float, default=16.0)
    ap.add_argument("--max-imgs", type=int, default=0, help="0=전체")
    ap.add_argument("--out", default="reports/sahi_bench")
    a = ap.parse_args()

    imgs = sorted(Path(a.img_dir).glob("*.jpg"))
    if a.max_imgs:
        imgs = imgs[:a.max_imgs]
    model = YOLO(a.weights)
    torch.cuda.empty_cache()

    out = {"weights": a.weights, "images": len(imgs), "modes": {}}
    for mode in ("base640", "hi1280", "tile640"):
        r = evaluate(mode, model, imgs, Path(a.lbl_dir), a)
        out["modes"][mode] = r
        print(f"{mode:8} far(<{a.far_px:g}px)={r['far_recall']} overall={r['overall_recall']} "
              f"FP/img={r['FP_per_image']} {r['ms_per_image']}ms  bins={r['recall_by_bin']}")

    Path(a.out + ".json").write_text(json.dumps(out, indent=2))
    L = ["# SAHI 타일/고해상 추론 벤치 — 재학습 없음", "",
         f"모델 고정 `{Path(a.weights).name}`, DUT-test {len(imgs)}장(네이티브 1280×720), "
         f"conf={a.conf}, IoU매칭 0.5. tile640 = 640타일(overlap {a.overlap}) + 전체프레임 + 병합 NMS.",
         "Maciullo-test 는 네이티브 640×480 → 타일링 무의미(단일 타일)라 제외.", "",
         "| mode | far(<%gpx) recall | overall recall | FP/img | ms/img(4090) | " % a.far_px
         + " | ".join(BIN_LABELS) + " |",
         "|---|---:|---:|---:|---:|" + "---:|" * len(BIN_LABELS)]
    for mode, r in out["modes"].items():
        L.append(f"| {mode} | **{r['far_recall']}** | {r['overall_recall']} | {r['FP_per_image']} | "
                 f"{r['ms_per_image']} | " + " | ".join(str(x) for x in r["recall_by_bin"]) + " |")
    L += ["", f"(bin GT 수: {out['modes']['base640']['gt_total']}, far GT {out['modes']['base640']['far_gt']})"]
    Path(a.out + ".md").write_text("\n".join(L) + "\n")
    print(f"\nsaved {a.out}.md / .json")


if __name__ == "__main__":
    main()
