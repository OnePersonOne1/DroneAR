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

| Split | Images | XML | Notes |
|-------|-------:|----:|-------|
| train | 5200 | 5200 | _filled in Phase 1_ |
| val   | 2600 | 2600 | _filled in Phase 1_ |
| test  | 2200 | 2200 | _filled in Phase 1_ |

- Single class: source label `UAV` → mapped to class `0: drone` (`nc=1`).
- Box-size distribution (small-object share) → see Phase 1. _placeholder_

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
