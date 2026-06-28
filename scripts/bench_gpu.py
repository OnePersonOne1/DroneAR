#!/usr/bin/env python3
"""GPU(CUDA) forward latency 벤치 — YOLO26 .pt 가중치.

측정 하드웨어의 순수 forward latency를 측정한다(전·후처리 제외). YOLO26은 NMS-free
one-to-one head라 forward 출력이 곧 `(1,300,6)`이며 NMS 단계가 없다.

프로토콜: imgsz=640, batch=1(single-stream), warmup≥30, iters≥200, fused 모델(배포 형태),
torch.cuda.synchronize 기반 정확 계측, mean±std(ms), FPS = 1000/mean.
"""
import argparse
import statistics
import time
from pathlib import Path

import torch
from ultralytics import YOLO


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="+",
                    default=["weights/yolo26n_drone_640.pt", "weights/yolo26s_drone_640.pt"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--precisions", nargs="+", default=["fp32", "fp16"])
    ap.add_argument("--out", default="weights/latency_gpu.md")
    return ap.parse_args()


@torch.no_grad()
def bench(model, x, warmup, iters):
    # cuda.Event로 순수 GPU 커널 시간을 계측(CPU 호출/sync 오버헤드 제외).
    for _ in range(warmup):
        model(x)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(iters):
        start.record()
        model(x)
        end.record()
        torch.cuda.synchronize()
        ts.append(start.elapsed_time(end))     # ms
    return statistics.mean(ts), (statistics.stdev(ts) if len(ts) > 1 else 0.0)


def main():
    a = parse_args()
    assert torch.cuda.is_available(), "CUDA 필요"
    dev = torch.device(a.device)
    gpu = torch.cuda.get_device_name(dev)

    rows = []
    for w in a.weights:
        stem = Path(w).stem
        for prec in a.precisions:
            half = prec == "fp16"
            m = YOLO(w).model.fuse().eval().to(dev)
            if half:
                m = m.half()
            x = torch.randn(1, 3, a.imgsz, a.imgsz, device=dev,
                            dtype=torch.half if half else torch.float32)
            mean, std = bench(m, x, a.warmup, a.iters)
            fps = 1000.0 / mean
            rows.append((stem, prec.upper(), mean, std, fps))
            print(f"{stem} {prec.upper()}: {mean:.3f} ± {std:.3f} ms  {fps:.1f} FPS")
            del m
            torch.cuda.empty_cache()

    out = Path(a.out)
    L = [f"# GPU latency — {gpu}", "",
         f"- imgsz={a.imgsz}, batch=1 (single-stream), warmup={a.warmup}, iters={a.iters}",
         "- 순수 forward(전·후처리·NMS 제외). NMS-free one-to-one head.",
         f"- device={a.device}, torch CUDA. FPS = 1000 / mean_latency_ms.", "",
         "| 모델 | 정밀도 | latency mean±std (ms) | FPS |", "|---|---|---:|---:|"]
    for stem, prec, mean, std, fps in rows:
        L.append(f"| {stem} | {prec} | {mean:.3f} ± {std:.3f} | {fps:.1f} |")
    L += ["",
          "INT8(GPU): TensorRT 엔진 빌드 시에만 측정. 미빌드 → TODO. 리포의 INT8 ONNX는 "
          "CPU/XNNPACK용 QDQ Conv-only라 CUDA EP에서 비대표."]
    out.write_text("\n".join(L) + "\n")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
