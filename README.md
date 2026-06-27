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
| Train | `... scripts/train.py` | `python scripts/train.py` |
| Evaluate | _Phase 3_ | _Phase 3_ |
| Export (ONNX/FP16/INT8) | _Phase 4_ | _Phase 4_ |
| Latency bench | _Phase 4_ | _Phase 4_ |

---

## Results

### Accuracy (val/test)

| Model | imgsz | mAP50 | mAP50-95 | Precision | Recall |
|-------|------:|------:|---------:|----------:|-------:|
| yolo26n | 640 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| yolo26s | 640 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

### Export precision (FP32 vs FP16 vs INT8)

| Model | Precision | File | Size | mAP50 / Δ |
|-------|-----------|------|-----:|----------:|
| _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

### Dev-CPU latency (directional estimate, **not** ML2)

> ⚠️ Numbers below are ONNX Runtime / XNNPACK on an **x86-64 dev CPU**. They are a
> *directional* proxy for the ML2 Zen2 mobile CPU. Final figures require on-device
> ML2 profiling over ADB.

| Model | Precision | threads=1 (ms) | threads=4 (ms) |
|-------|-----------|---------------:|---------------:|
| _TBD_ | _TBD_ | _TBD_ | _TBD_ |

---

## Magic Leap 2 deployment (next-step guide)

_Filled in Phase 5._ Recommended path: ONNX (opset17, NMS-free) → ONNX Runtime (+MLSDK C API),
CPU backend XNNPACK. Output tensor `(1,300,6)` = `[x1,y1,x2,y2,score,class]` (one-to-one head),
so **no device-side NMS**. App pipeline: ML2 camera frame → preprocess (resize 640, /255, CHW)
→ ORT infer → score threshold → rescale boxes to native resolution → AR overlay.

---

## License / notes

Dataset (DUT-Anti-UAV) retains its own license; not redistributed here.
