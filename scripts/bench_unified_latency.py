#!/usr/bin/env python3
"""전 모델 통일 latency 벤치 — yolo·D-FINE 동일 harness(torch forward).

공정 비교: 같은 하드웨어(4090 GPU · Ryzen CPU)·같은 계측(순수 forward, 전·후처리 제외)·
각 모델 배포 imgsz. GPU=cuda.Event, CPU=wall-clock. batch=1.
- yolo: ultralytics YOLO(w).model.fuse() forward.
- dfine: dfine_eval.load_model() 의 deploy 모델 forward(postprocessor 제외 = yolo forward와 동급).

산출: reports/unified_latency.json (+ 콘솔 표).
usage: python scripts/bench_unified_latency.py
"""
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

# (key, 표시명, family, imgsz, weights, [config])
MODELS = [
    ("yolo26n_m300", "yolo26n (merged)", "yolo", 640,
     "weights/yolo26/yolo26n_drone_640_mergedataset_300epoch.pt", None),
    ("yolo26s_640",  "yolo26s",          "yolo", 640,
     "weights/yolo26/yolo26s_drone_640.pt", None),
    ("yolo26lP2",    "yolo26l-P2 (merged)", "yolo", 960,
     "weights/yolo26/yolo26l_drone_960p2_mergedataset_100epoch.pt", None),
    ("dfine_n",      "D-FINE-N (merged)", "dfine", 640,
     "weights/d_fine/dfine_n_drone_640_mergedataset_220epoch.pth",
     "configs/dfine/custom/dfine_n640_merged.yml"),
    ("dfine_l",      "D-FINE-L (merged)", "dfine", 960,
     "/workspace/runs/merged_dfine_l_960/best_stg2.pth",
     "configs/dfine/custom/dfine_l960_merged.yml"),
]
DFINE_ROOT = "/root/D-FINE"


def load_forward(m, device, half=False):
    """(callable forward, input tensor) 반환."""
    key, name, fam, imgsz, w, cfg = m
    if fam == "yolo":
        from ultralytics import YOLO
        net = YOLO(w).model.fuse().eval().to(device)
        if half:
            net = net.half()
    else:
        from dfine_eval import load_model
        net, _post = load_model(DFINE_ROOT, cfg, w, (imgsz, imgsz), str(device))
        if half:
            net = net.half()
    x = torch.randn(1, 3, imgsz, imgsz, device=device,
                    dtype=torch.half if half else torch.float32)
    return net, x


@torch.no_grad()
def bench_gpu(net, x, warmup=20, iters=100):
    for _ in range(warmup):
        net(x)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(iters):
        s.record(); net(x); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.mean(ts), (statistics.stdev(ts) if len(ts) > 1 else 0.0)


@torch.no_grad()
def bench_cpu(net, x, warmup=3, iters=12):
    for _ in range(warmup):
        net(x)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter(); net(x); ts.append((time.perf_counter() - t0) * 1000)
    return statistics.mean(ts), (statistics.stdev(ts) if len(ts) > 1 else 0.0)


def main():
    assert torch.cuda.is_available(), "CUDA 필요"
    gpu = torch.cuda.get_device_name(0)
    import platform
    results = {}
    for m in MODELS:
        key, name, fam, imgsz, w, cfg = m
        r = {"name": name, "family": fam, "imgsz": imgsz}
        # --- GPU fp32/fp16 ---
        for half in (False, True):
            tag = "gpu_fp16" if half else "gpu_fp32"
            try:
                net, x = load_forward(m, torch.device("cuda:0"), half=half)
                mean, std = bench_gpu(net, x)
                r[tag] = {"ms": round(mean, 3), "std": round(std, 3), "fps": round(1000/mean, 1)}
                print(f"{name:22} {tag}: {mean:.3f}±{std:.3f} ms  {1000/mean:.1f} FPS")
                del net, x; torch.cuda.empty_cache()
            except Exception as ex:
                r[tag] = {"error": str(ex)[:120]}
                print(f"{name:22} {tag}: ERROR {str(ex)[:80]}")
                torch.cuda.empty_cache()
        # --- CPU fp32 (threads 1, 8) ---
        for th in (1, 8):
            tag = f"cpu_t{th}"
            try:
                torch.set_num_threads(th)
                net, x = load_forward(m, torch.device("cpu"), half=False)
                mean, std = bench_cpu(net, x)
                r[tag] = {"ms": round(mean, 2), "std": round(std, 2), "fps": round(1000/mean, 2)}
                print(f"{name:22} {tag}: {mean:.2f}±{std:.2f} ms  {1000/mean:.2f} FPS")
                del net, x
            except Exception as ex:
                r[tag] = {"error": str(ex)[:120]}
                print(f"{name:22} {tag}: ERROR {str(ex)[:80]}")
        results[key] = r

    meta = {"gpu": gpu, "cpu": platform.processor() or "AMD Ryzen 9 7950X",
            "note": "순수 forward(전·후처리 제외), batch=1, 각 모델 배포 imgsz. "
                    "GPU=cuda.Event(warmup20/iter100), CPU=wall-clock(warmup3/iter12)."}
    Path("reports").mkdir(exist_ok=True)
    Path("reports/unified_latency.json").write_text(json.dumps({"meta": meta, **results}, indent=2))
    print("\nsaved reports/unified_latency.json")


if __name__ == "__main__":
    main()
