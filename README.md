# DroneAR — YOLO26 Drone Detection for Magic Leap 2

End-to-end, reproducible pipeline to train a **YOLO26** drone (UAV) object detector on the
**DUT-Anti-UAV** dataset and export it to a form deployable on **Magic Leap 2 (ML2)**.

- **Train target:** RTX 4090 24GB / Linux / CUDA (training only).
- **Inference target:** Magic Leap 2 — AMD "Mero" SoC (Zen2 quad-core x86-64 CPU + RDNA2 iGPU),
  16GB, AOSP Android 10 (API 29). **Not NVIDIA** → no TensorRT/CUDA on device.
  Validated on-device path: **ONNX → ONNX Runtime (+MLSDK C API), CPU backend XNNPACK.**
- **Model decision:** `yolo26n` (nano) first, **NMS-free one-to-one head kept**, `imgsz=640`,
  INT8 / FP16 export. CPU inference keeps the RDNA2 GPU free for 120Hz AR stereo rendering.

> Status: living document — tables are filled in as each pipeline phase completes.

---

## Repository layout

```
scripts/      voc2yolo.py · train.py · predict.py · export.py · bench_latency.py
configs/      dut_drone.yaml
weights/      best.pt(s) · *.onnx (fp32/fp16/int8) · *.tflite · metrics.json · latency_report.md
Dockerfile · docker-compose.yml · .dockerignore · requirements.txt · README.md
```

---

## Dataset (manual prep — shared, read-only)

DUT-Anti-UAV is **not** committed. Place/extract it at `/mnt/ssd_0/dataset/DUT` with the
PASCAL VOC structure below (the converter is read-only w.r.t. this tree):

```
/mnt/ssd_0/dataset/DUT/{train,val,test}/{img,xml}
  img/  *.jpg
  xml/  *.xml   (VOC: <size>, <object><name>, <bndbox> xmin/ymin/xmax/ymax)
```

| Split | Images | Labels | Boxes | Negatives | Skipped (degenerate) |
|-------|-------:|-------:|------:|----------:|---------------------:|
| train | 5200 | 5200 | 5243 | 3 | 0 |
| val   | 2600 | 2600 | 2620 | 0 | 1 |
| test  | 2200 | 2200 | 2245 | 0 | 0 |
| **total** | **10000** | **10000** | **10108** | **3** | **1** |

- Single class: source label `UAV` (10,109 objects) → mapped to class `0: drone` (`nc=1`).
- 3 train images have no object → emitted as empty `.txt` negatives. 1 degenerate box (w≤0/h≤0) skipped.

**Convert (read-only w.r.t. source):**
```bash
python scripts/voc2yolo.py        # --src /mnt/ssd_0/dataset/DUT  --dst /mnt/ssd_0/dataset/dut_yolo
python scripts/dataset_stats.py   # box-size histogram + sample box overlays -> dut_yolo/_viz/
```

**Box-size distribution — small-object-heavy** (drives the imgsz/P2 decision):
normalized side `sqrt(w·h)`: median **0.0226** (~14.5px @640), p25 0.0163, p75 0.0451, max 0.84.

| Size bin (@imgsz 640) | Share |
|---|---:|
| SMALL (<32px side) | **76.6%** |
| MEDIUM (32–96px) | 13.1% |
| LARGE (>96px side) | 10.3% |
| tiny (<13px, norm side <0.02) | 40.6% |

→ Most drones are small/tiny. Baseline stays `imgsz=640` (ML2 target), but **imgsz=960 and a P2
head are flagged as the primary accuracy levers** for small-object recall (Phase 2 step 3).

---

## Environment

### Option A — Docker (recommended, collaborator-reproducible)

```bash
docker compose build
docker compose run --rm dronear python scripts/voc2yolo.py
docker compose run --rm dronear python scripts/train.py
docker compose run --rm dronear python scripts/export.py
```

The shared dataset is mounted host-path → identical container-path, so `configs/dut_drone.yaml`
works unchanged in both native and container runs. On another machine, edit the dataset volume
in `docker-compose.yml` and the `path:` line in the config.

**Reproducibility verified:** `docker compose build` (base `ultralytics/ultralytics:latest` +
`onnxruntime`/`onnxslim`/`onnxconverter-common`, with stock polars swapped for `polars-lts-cpu`)
produces a working GPU image (CUDA reachable in-container). Running
`docker compose run --rm dronear python scripts/export.py --weights weights/yolo26n_drone_640.pt
--stem yolo26n_drone_640 --outdir weights/docker_verify` inside the container produced the same
artifacts as the host venv (FP32 9.80 MB, FP16 4.97 MB native-half, INT8 3.01 MB), all loading in
ORT with output `[1,300,6]`.

### Option B — venv (fast dev loop)

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt           # or: pip install -U "ultralytics>=8.4.0" onnx onnxruntime onnxslim
python scripts/voc2yolo.py
python scripts/train.py
```

---

## Reproduce (full commands)

_Filled across phases. Each step has a Docker and a venv form._

| Step | Docker | venv |
|------|--------|------|
| Convert VOC→YOLO | `docker compose run --rm dronear python scripts/voc2yolo.py` | `python scripts/voc2yolo.py` |
| Dataset stats | `... python scripts/dataset_stats.py` | `python scripts/dataset_stats.py` |
| Train (single) | `... python scripts/train.py --model yolo26n.pt --name yolo26n_drone_640` | `python scripts/train.py ...` |
| Train (n+s, 150ep) | `... bash scripts/train_all.sh` | `bash scripts/train_all.sh` |
| Evaluate (val+test) | `... python scripts/eval.py --weights weights/yolo26n_drone_640.pt` | `python scripts/eval.py ...` |
| Export ONNX/FP16/INT8 | `... python scripts/export.py --weights weights/yolo26n_drone_640.pt --stem yolo26n_drone_640` | `python scripts/export.py ...` |
| Latency bench | `... python scripts/bench_latency.py --stem yolo26n_drone_640` | `python scripts/bench_latency.py ...` |
| Predict demo | `... python scripts/predict.py --weights weights/yolo26n_drone_640.pt` | `python scripts/predict.py ...` |

Optional TFLite INT8 (alternative ML2 path via NNAPI/XNNPACK) — requires a TensorFlow/onnx2tf
toolchain (not installed here): `python -c "from ultralytics import YOLO;
YOLO('weights/yolo26n_drone_640.pt').export(format='tflite', int8=True,
data='configs/dut_drone.yaml', imgsz=640)"`.

**Training config (ML2 baseline):** `yolo26n.pt`, `imgsz=640`, `epochs=150`, `patience=40`,
`batch=-1` (auto → ~35 on the 4090), `cache=disk`, NMS-free one-to-one head kept. `yolo26s` is
the accuracy comparison. Quick 5-epoch smoke confirmed convergence (mAP50 0.62→0.81 in 5 epochs).

### Troubleshooting (environment fixes, baked into requirements)

| Symptom | Cause | Fix |
|---|---|---|
| `cuda.is_available()=False`, "driver too old" | ultralytics pulls torch `cu130`; this host's driver is CUDA 12.8 | install `torch==2.11.0+cu128` (newest cu128 build) |
| **Bus error (SIGBUS)** at first checkpoint save | `polars` 1.42 wheel SIGBUS on import on this CPU; ultralytics reads `results.csv` via polars every epoch | replace with **`polars-lts-cpu`** |
| SIGBUS with `cache=ram` | DataLoader shares cached arrays via `/dev/shm` | use `cache=disk` (default) or `--cache False` |

---

## Results

### Accuracy (150 epochs, imgsz 640) — from `weights/metrics.json`

| Model | Split | mAP50 | mAP50-95 | Precision | Recall | params | best.pt |
|-------|-------|------:|---------:|----------:|-------:|-------:|--------:|
| **yolo26n** (ML2 primary) | val | 0.911 | 0.583 | 0.958 | 0.872 | 2.4M | 5.4 MB |
| yolo26n | test | **0.951** | 0.648 | 0.963 | 0.922 | | |
| yolo26s (comparison) | val | 0.929 | 0.617 | 0.963 | 0.903 | 9.5M | 20.3 MB |
| yolo26s | test | **0.958** | 0.681 | 0.968 | 0.945 | | |

yolo26s gives ~+0.7pp test mAP50 / ~+3pp mAP50-95 over yolo26n for ~4× params and ~4× GFLOPs
(5.2→20.5). For the ML2 CPU target, **yolo26n is the recommended deploy model**; yolo26s is the
accuracy ceiling if latency budget allows. Example detection (tiny drone, conf 0.78):
`docs/demo/`.

### Export precision — yolo26n (ML2 primary), imgsz 640, NMS-free head, output `[1,300,6]`

| Precision | File | Size | Notes |
|-----------|------|-----:|-------|
| FP32 | `weights/yolo26n_drone_640_fp32.onnx` | 9.80 MB | reference; opset17, static, simplified |
| FP16 | `weights/yolo26n_drone_640_fp16.onnx` | 4.97 MB | native `half=True`; float16 I/O |
| INT8 | `weights/yolo26n_drone_640_int8.onnx` | **3.01 MB** | static PTQ (QDQ), Conv-only, 200-img calib |

**INT8 fidelity vs FP32** (same 20 val images, conf 0.25): detections **27 → 27**, all matched
at IoU≥0.5, mean IoU 0.961, mean |Δscore| 0.075 → negligible accuracy loss.

### Dev-CPU latency (directional estimate, **not** ML2) — `weights/latency_report.md`

> ⚠️ ONNX Runtime / CPUExecutionProvider on an **x86-64 desktop CPU** (i9-13900K). A
> *directional* proxy for the ML2 Zen2 mobile CPU, **not** a measurement of it. Final
> figures require on-device ML2 profiling over ADB (the device uses ORT + XNNPACK).

| Precision | threads=1 (ms) | threads=4 (ms) | Size |
|-----------|---------------:|---------------:|-----:|
| FP32 | 41.9 ± 1.5 | 12.9 ± 0.5 | 9.80 MB |
| FP16 | 42.9 ± 0.9 | 13.4 ± 0.3 | 4.97 MB |
| INT8 | **30.2 ± 0.9** | 14.1 ± 0.4 | **3.01 MB** |

INT8 wins on size and single-thread latency; at 4 threads the Conv-only QDQ dequant overhead
narrows the gap on x86 (XNNPACK on ML2 behaves differently). FP16 gives no CPU speedup (ORT
CPU lacks native fp16 kernels) — it is a size/portability option.

---

## Magic Leap 2 deployment (next-step guide)

_Filled in Phase 5._ Recommended path: ONNX (opset17, NMS-free) → ONNX Runtime (+MLSDK C API),
CPU backend XNNPACK. Output tensor `(1,300,6)` = `[x1,y1,x2,y2,score,class]` (one-to-one head),
so **no device-side NMS**. App pipeline: ML2 camera frame → preprocess (resize 640, /255, CHW)
→ ORT infer → score threshold → rescale boxes to native resolution → AR overlay.

---

## License / notes

Dataset (DUT-Anti-UAV) retains its own license; not redistributed here.
