#!/usr/bin/env python3
"""CPU latency 벤치 + FP32-vs-INT8 정확도 점검 — 내보낸 ONNX 모델.

ONNX Runtime CPUExecutionProvider에서 intra_op_num_threads = 1, 4로 측정한다.
warmup 제외 mean/std(ms)를 보고한다. FPS = 1000 / mean.

본 스크립트는 데스크톱 CPU 실측만 수행한다(추정/기기 비교 없음).

출력: weights/latency_report.md
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
    ap.add_argument("--stems", nargs="+",
                    default=["yolo26n_drone_640", "yolo26s_drone_640"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--calib-dir", default="/mnt/ssd_0/dataset/dut_yolo/images/val")
    ap.add_argument("--conf", type=float, default=0.25)
    return ap.parse_args()


def cpu_name():
    try:
        for l in open("/proc/cpuinfo"):
            if l.startswith("model name"):
                return l.split(":", 1)[1].strip()
    except Exception:
        pass
    import platform
    return platform.processor() or "unknown CPU"


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
        a = a[0]
    return a[a[:, 4] >= conf]


def iou(b1, b2):
    xa, ya = max(b1[0], b2[0]), max(b1[1], b2[1])
    xb, yb = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    a1 = max(0, b1[2] - b1[0]) * max(0, b1[3] - b1[1])
    a2 = max(0, b2[2] - b2[0]) * max(0, b2[3] - b2[1])
    return inter / (a1 + a2 - inter + 1e-9)


def compare_fp32_int8(fp32, int8, calib_dir, imgsz, conf, n=20):
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
        for a in d32:
            if len(d8) == 0:
                continue
            j = max(range(len(d8)), key=lambda k: iou(a[:4], d8[k][:4]))
            if iou(a[:4], d8[j][:4]) >= 0.5:
                matched += 1
                score_diffs.append(abs(a[4] - d8[j][4]))
                ious.append(iou(a[:4], d8[j][:4]))
    return {"images": len(files), "det_fp32": dn32, "det_int8": dn8, "matched": matched,
            "mean_score_diff": float(np.mean(score_diffs)) if score_diffs else float("nan"),
            "mean_iou": float(np.mean(ious)) if ious else float("nan")}


def main():
    a = parse_args()
    wd = Path(a.weights_dir)
    cpu = cpu_name()

    # 측정 + 콘솔 출력.
    table = []   # (stem, precision, size, {t: (mean,std)})
    cmps = []    # (stem, cmp dict)
    for stem in a.stems:
        variants = [(p, wd / f"{stem}_{p.lower()}.onnx") for p in ("FP32", "FP16", "INT8")]
        variants = [(n, p) for n, p in variants if p.exists()]
        for name, path in variants:
            size = path.stat().st_size / 1e6
            cells = {t: bench(path, a.imgsz, a.runs, a.warmup, t) for t in a.threads}
            table.append((stem, name, size, cells))
            print(f"{stem} {name}: size={size:.2f}MB " +
                  " ".join(f"t{t}={cells[t][0]:.2f}±{cells[t][1]:.2f}ms({1000/cells[t][0]:.0f}fps)"
                           for t in a.threads))
        fp32p, int8p = wd / f"{stem}_fp32.onnx", wd / f"{stem}_int8.onnx"
        if fp32p.exists() and int8p.exists():
            c = compare_fp32_int8(fp32p, int8p, a.calib_dir, a.imgsz, a.conf)
            cmps.append((stem, c))
            print(f"  {stem} FP32 vs INT8: det {c['det_fp32']}->{c['det_int8']} "
                  f"matched={c['matched']} mean|Δscore|={c['mean_score_diff']:.4f} "
                  f"mean_IoU={c['mean_iou']:.4f}")

    # 리포트.
    out = wd / "latency_report.md"
    th = a.threads
    L = [f"# CPU latency — {cpu}", "",
         "데스크톱 CPU 실측(추정/기기 비교 없음).", "",
         f"- ONNX Runtime CPUExecutionProvider, imgsz={a.imgsz}, batch=1 (single-stream)",
         f"- warmup={a.warmup}, iters={a.runs}, intra_op_num_threads={th} (inter_op=1, sequential)",
         "- FPS = 1000 / mean_latency_ms", "",
         "| 모델 | 정밀도 | 크기(MB) | " +
         " | ".join(f"t={t} mean±std(ms)" for t in th) + " | " +
         " | ".join(f"t={t} FPS" for t in th) + " |",
         "|---|---|---:|" + "---:|" * (2 * len(th))]
    for stem, name, size, cells in table:
        ms = " | ".join(f"{cells[t][0]:.2f} ± {cells[t][1]:.2f}" for t in th)
        fps = " | ".join(f"{1000/cells[t][0]:.0f}" for t in th)
        L.append(f"| {stem} | {name} | {size:.2f} | {ms} | {fps} |")
    for stem, c in cmps:
        L += ["", f"## FP32 vs INT8 정확도 — {stem} (동일 이미지 {c['images']}장, conf={a.conf})", "",
              f"- 탐지: FP32 **{c['det_fp32']}** vs INT8 **{c['det_int8']}**, 매칭(IoU≥0.5) {c['matched']}",
              f"- 평균 |Δscore| {c['mean_score_diff']:.4f}, 평균 IoU {c['mean_iou']:.4f}"]
    L += ["", "## Notes",
          "- **FP16**: ORT CPU에 native fp16 커널이 없어 up/down-cast → CPU 속도 이득 없음(크기/이식성 옵션).",
          "- **INT8**: Conv 레이어만 양자화(QDQ). NMS-free detection head는 float 유지 → 작은 score가 0으로 뭉개지지 않음."]
    out.write_text("\n".join(L) + "\n")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
