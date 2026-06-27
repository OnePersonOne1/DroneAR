#!/usr/bin/env python3
"""Latency benchmark + FP32-vs-INT8 accuracy check for the exported ONNX models.

Runs ONNX Runtime on CPU (CPUExecutionProvider) at intra_op_num_threads = 1 and 4
(4 ~= the ML2 Zen2 quad-core). Reports warmup-excluded mean/std over N runs.

!!! IMPORTANT: these numbers come from an x86-64 desktop CPU. They are a DIRECTIONAL
proxy for the Magic Leap 2 Zen2 mobile CPU, NOT a substitute. Final figures require
on-device ML2 profiling over ADB. !!!

Output: weights/latency_report.md
"""
import argparse
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights-dir", default="weights")
    ap.add_argument("--stem", default="yolo26n_drone_640")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--calib-dir", default="/mnt/ssd_0/dataset/dut_yolo/images/val")
    ap.add_argument("--conf", type=float, default=0.25)
    return ap.parse_args()


def letterbox(img, new=640, color=114):
    h, w = img.shape[:2]
    r = min(new / h, new / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new, new, 3), color, dtype=np.uint8)
    top, left = (new - nh) // 2, (new - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.transpose(rgb, (2, 0, 1))[None]


def make_session(path, threads):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(str(path), sess_options=so,
                                providers=["CPUExecutionProvider"])


def bench(path, imgsz, runs, warmup, threads):
    sess = make_session(path, threads)
    iname = sess.get_inputs()[0].name
    itype = sess.get_inputs()[0].type
    x = np.random.rand(1, 3, imgsz, imgsz).astype(
        np.float16 if "float16" in itype else np.float32)
    for _ in range(warmup):
        sess.run(None, {iname: x})
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {iname: x})
        ts.append((time.perf_counter() - t0) * 1000.0)
    return statistics.mean(ts), (statistics.stdev(ts) if len(ts) > 1 else 0.0)


def decode(out, conf):
    """NMS-free one-to-one head -> (N,6) [x1,y1,x2,y2,score,class] above conf."""
    a = np.array(out[0])
    if a.ndim == 3:
        a = a[0]                      # (300,6)
    return a[a[:, 4] >= conf]


def iou(b1, b2):
    xa, ya = max(b1[0], b2[0]), max(b1[1], b2[1])
    xb, yb = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    a1 = max(0, b1[2] - b1[0]) * max(0, b1[3] - b1[1])
    a2 = max(0, b2[2] - b2[0]) * max(0, b2[3] - b2[1])
    return inter / (a1 + a2 - inter + 1e-9)


def compare_fp32_int8(fp32, int8, calib_dir, imgsz, conf, n=20):
    """Run both on the same images; report detection-count and score agreement."""
    s32 = make_session(fp32, 4); s8 = make_session(int8, 4)
    n32, n8 = s32.get_inputs()[0].name, s8.get_inputs()[0].name
    files = sorted(p for p in Path(calib_dir).iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png"))[:n]
    dn32 = dn8 = matched = 0
    score_diffs, ious = [], []
    for f in files:
        x = letterbox(cv2.imread(str(f)), imgsz)
        d32 = decode(s32.run(None, {n32: x}), conf)
        d8 = decode(s8.run(None, {n8: x}), conf)
        dn32 += len(d32); dn8 += len(d8)
        for a in d32:                       # greedy match by IoU
            if len(d8) == 0:
                continue
            j = max(range(len(d8)), key=lambda k: iou(a[:4], d8[k][:4]))
            if iou(a[:4], d8[j][:4]) >= 0.5:
                matched += 1
                score_diffs.append(abs(a[4] - d8[j][4]))
                ious.append(iou(a[:4], d8[j][:4]))
    return {
        "images": len(files), "det_fp32": dn32, "det_int8": dn8, "matched": matched,
        "mean_score_diff": float(np.mean(score_diffs)) if score_diffs else float("nan"),
        "mean_iou": float(np.mean(ious)) if ious else float("nan"),
    }


def main():
    a = parse_args()
    wd = Path(a.weights_dir)
    variants = [("FP32", wd / f"{a.stem}_fp32.onnx"),
                ("FP16", wd / f"{a.stem}_fp16.onnx"),
                ("INT8", wd / f"{a.stem}_int8.onnx")]
    variants = [(n, p) for n, p in variants if p.exists()]

    rows = []
    for name, path in variants:
        size = path.stat().st_size / 1e6
        cells = {}
        for t in a.threads:
            m, s = bench(path, a.imgsz, a.runs, a.warmup, t)
            cells[t] = (m, s)
        rows.append((name, path.name, size, cells))
        print(f"{name}: size={size:.2f}MB " +
              " ".join(f"t{t}={cells[t][0]:.2f}±{cells[t][1]:.2f}ms" for t in a.threads))

    # Accuracy: FP32 vs INT8.
    cmp = None
    fp32p, int8p = wd / f"{a.stem}_fp32.onnx", wd / f"{a.stem}_int8.onnx"
    if fp32p.exists() and int8p.exists():
        cmp = compare_fp32_int8(fp32p, int8p, a.calib_dir, a.imgsz, a.conf)
        print(f"\nFP32 vs INT8: det {cmp['det_fp32']}->{cmp['det_int8']} "
              f"matched={cmp['matched']} mean|Δscore|={cmp['mean_score_diff']:.4f} "
              f"mean_IoU={cmp['mean_iou']:.4f}")

    # Report.
    out = wd / "latency_report.md"
    lines = [f"# Latency report — {a.stem}", "",
             "> **Directional estimate only.** ONNX Runtime / CPUExecutionProvider on an",
             "> x86-64 desktop CPU. This is a proxy for the Magic Leap 2 Zen2 mobile CPU,",
             "> not a measurement of it. Final numbers require on-device ML2 ADB profiling.",
             "",
             f"- imgsz={a.imgsz}, runs={a.runs} (warmup={a.warmup}), batch=1",
             f"- threads tested: {a.threads} (4 ≈ ML2 Zen2 quad-core)", "",
             "| Precision | File | Size (MB) | " +
             " | ".join(f"t={t} mean±std (ms)" for t in a.threads) + " |",
             "|---|---|---:|" + "---:|" * len(a.threads)]
    for name, fn, size, cells in rows:
        lines.append(f"| {name} | `{fn}` | {size:.2f} | " +
                     " | ".join(f"{cells[t][0]:.2f} ± {cells[t][1]:.2f}" for t in a.threads) + " |")
    if cmp:
        lines += ["", "## FP32 vs INT8 accuracy (same images, conf="
                  f"{a.conf})", "",
                  f"- images: {cmp['images']}",
                  f"- detections: FP32 **{cmp['det_fp32']}** vs INT8 **{cmp['det_int8']}**",
                  f"- matched (IoU≥0.5): {cmp['matched']}",
                  f"- mean |Δscore| on matches: {cmp['mean_score_diff']:.4f}",
                  f"- mean IoU on matches: {cmp['mean_iou']:.4f}"]
    out.write_text("\n".join(lines) + "\n")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
