#!/usr/bin/env python3
"""전 모델 통일 latency 벤치 — yolo·D-FINE 동일 harness(torch forward).

공정 비교: 같은 하드웨어(4090 GPU · Ryzen CPU)·같은 계측(순수 forward, 전·후처리 제외)·
각 모델 배포 imgsz. GPU=cuda.Event, CPU=wall-clock. batch=1.
- yolo: ultralytics YOLO(w).model.fuse() forward.
- dfine: dfine_eval.load_model() 의 deploy 모델 forward(postprocessor 제외 = yolo forward와 동급).

precision:
- GPU fp32: 기본.
- GPU fp16: yolo = 전체 half(.half()); dfine = torch.autocast(AMP) — grid_sample은 fp32 유지되어
  이득 작음(실 fp16 가속은 TensorRT 엔진). 계열별 fp16 배포 형태와 일치.
- CPU: fp32만(ORT fp16 native 커널 없음).

⚠️ D-FINE 모델을 한 프로세스에서 2개 로드하면 D-FINE config 레지스트리 상태가 bleed되어
   두 번째 로드가 잘못된 차원으로 빌드됨 → **모델별 subprocess 격리**로 실행(기본 동작).

산출: reports/unified_latency.json. usage: python scripts/bench_unified_latency.py
       (내부: python scripts/bench_unified_latency.py --only <key>)
"""
import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

# (key, 표시명, family, imgsz, weights, [config])
MODELS = [
    ("yolo26n_m300", "yolo26n (merged)", "yolo", 640,
     "weights/yolo26/yolo26n_drone_640_mergedataset_300epoch.pt", None),
    ("yolo26s_640", "yolo26s", "yolo", 640,
     "weights/yolo26/yolo26s_drone_640.pt", None),
    ("yolo26lP2", "yolo26l-P2 (merged)", "yolo", 960,
     "weights/yolo26/yolo26l_drone_960p2_mergedataset_100epoch.pt", None),
    ("dfine_n", "D-FINE-N (merged)", "dfine", 640,
     "weights/d_fine/dfine_n_drone_640_mergedataset_220epoch.pth",
     "configs/dfine/custom/dfine_n640_merged.yml"),
    ("dfine_l", "D-FINE-L (merged)", "dfine", 960,
     "/workspace/runs/merged_dfine_l_960/best_stg2.pth",
     "configs/dfine/custom/dfine_l960_merged.yml"),
]
DFINE_ROOT = "/root/D-FINE"


def load_model_fp32(m, device):
    key, name, fam, imgsz, w, cfg = m
    if fam == "yolo":
        from ultralytics import YOLO
        net = YOLO(w).model.fuse().eval().to(device)
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from dfine_eval import load_model
        net, _ = load_model(DFINE_ROOT, cfg, w, (imgsz, imgsz), str(device))
    return net


def make_x(imgsz, device, half=False):
    import torch
    return torch.randn(1, 3, imgsz, imgsz, device=device,
                       dtype=torch.half if half else torch.float32)


def _bench(net, x, warmup, iters, gpu, autocast=False):
    import torch
    ctx = torch.autocast("cuda", dtype=torch.float16) if autocast else _null()
    with torch.no_grad(), ctx:
        for _ in range(warmup):
            net(x)
        if gpu:
            torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            ts = []
            for _ in range(iters):
                s.record(); net(x); e.record(); torch.cuda.synchronize()
                ts.append(s.elapsed_time(e))
        else:
            ts = []
            for _ in range(iters):
                t0 = time.perf_counter(); net(x); ts.append((time.perf_counter() - t0) * 1000)
    return statistics.mean(ts), (statistics.stdev(ts) if len(ts) > 1 else 0.0)


class _null:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def bench_one(m):
    import torch
    key, name, fam, imgsz, w, cfg = m
    r = {"name": name, "family": fam, "imgsz": imgsz}
    dev = torch.device("cuda:0")
    # GPU fp32
    net = load_model_fp32(m, dev); x = make_x(imgsz, dev)
    mean, std = _bench(net, x, 20, 100, gpu=True)
    r["gpu_fp32"] = {"ms": round(mean, 3), "std": round(std, 3), "fps": round(1000/mean, 1)}
    del net, x; torch.cuda.empty_cache()
    # fp16 미측정 — 전 모델 fp32 통일 비교(계열별 fp16 배포 스택 상이:
    #   yolo=ONNX/TRT half, D-FINE=TensorRT fp16(grid_sample 플러그인) → 공정 비교 불가).
    # CPU fp32 (threads 1, 8)
    for th in (1, 8):
        torch.set_num_threads(th)
        net = load_model_fp32(m, torch.device("cpu")); x = make_x(imgsz, torch.device("cpu"))
        mean, std = _bench(net, x, 3, 12, gpu=False)
        r[f"cpu_t{th}"] = {"ms": round(mean, 2), "std": round(std, 2), "fps": round(1000/mean, 2)}
        del net, x
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="단일 model_key만 측정 → JSON stdout(내부용)")
    a = ap.parse_args()
    import torch
    assert torch.cuda.is_available(), "CUDA 필요"

    if a.only:
        m = [x for x in MODELS if x[0] == a.only][0]
        print("__RESULT__" + json.dumps(bench_one(m)))
        return

    # 모델별 subprocess 격리(D-FINE config bleed 방지)
    results = {}
    for m in MODELS:
        key = m[0]
        print(f"=== {m[1]} ===")
        p = subprocess.run([sys.executable, __file__, "--only", key],
                           capture_output=True, text=True)
        line = [l for l in p.stdout.splitlines() if l.startswith("__RESULT__")]
        if not line:
            print(p.stdout[-500:], p.stderr[-500:]); continue
        results[key] = json.loads(line[0][len("__RESULT__"):])
        r = results[key]
        print(f"  GPU fp32 {r['gpu_fp32']['fps']} FPS | CPU t8 {r['cpu_t8']['fps']} FPS")

    meta = {"gpu": torch.cuda.get_device_name(0), "cpu": platform.processor() or "AMD Ryzen 9 7950X",
            "precision": "fp32 (전 모델 통일)",
            "note": "순수 forward(전·후처리 제외), batch=1, 각 모델 배포 imgsz. "
                    "GPU=cuda.Event(warmup20/iter100), CPU=wall-clock(warmup3/iter12). "
                    "fp16 미측정 — 계열별 fp16 배포 스택 상이로 공정 비교 불가."}
    Path("reports").mkdir(exist_ok=True)
    Path("reports/unified_latency.json").write_text(
        json.dumps({"meta": meta, **results}, indent=2, ensure_ascii=False))
    print("saved reports/unified_latency.json")


if __name__ == "__main__":
    main()
