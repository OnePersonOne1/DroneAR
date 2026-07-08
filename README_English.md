# DroneAR — YOLO26 Drone Detection for Magic Leap 2

> One-page English summary. Full documentation (Korean): [README.md](README.md)

Train a **YOLO26** drone (UAV) detector on **DUT-Anti-UAV** (+ **Maciullo DroneDetectionDataset**
merged, 10× training data) → deploy to **Magic Leap 2** (on-device) and/or a cloud GPU.

## Setup

- **Training**: RTX 4090 / Linux / CUDA. Reproduce: `docker compose pull` (image `hanmyeongil/yolo26:v1`) or `pip install -r requirements.txt`.
- **Inference targets**: ML2 (AMD Zen2 x86-64 + RDNA2 iGPU, Android 10 / API 29 — no NVIDIA) via ONNX Runtime CPU or ncnn-Vulkan; cloud RTX 4090 via PyTorch.

## Models & results (fixed held-out test sets)

| model | config | DUT-test AP50 / AP50-95 | Maciullo-test AP50 / AP50-95 | use |
|---|---|---|---|---|
| `yolo26n_drone_640` | DUT-only, 150ep | 0.951 / 0.648 | 0.601 / 0.216 | legacy baseline |
| `..._mergedataset_300epoch` | merged, 640 | 0.950 / 0.650 | 0.858 / 0.415 | **on-device (ML2)** |
| `..._mergedataset_100epoch` | merged, 640 | 0.927 / 0.619 | 0.891 / 0.445 | Maciullo-domain focus |
| `yolo26n_drone_960p2_...` | merged, 960+P2 | 0.966 / 0.690 | 0.888 / 0.447 | cloud, lightweight |
| `yolo26l_drone_960p2_...` | merged, 960+P2, **l-scale** | **0.982 / 0.769** | 0.888 / **0.450** | **cloud (infer at 1280)** |

- Ablation (checkbox × AP, attribution limits): [reports/ablation_matrix.md](reports/ablation_matrix.md)
- Far/small-object analysis & resolution sweep: [reports/far_drone_p2_960.md](reports/far_drone_p2_960.md)
- YOLO26 n–x FPS on 4090: [reports/yolo26_family_fps_4090.md](reports/yolo26_family_fps_4090.md)

## ML2 deployment (ncnn-Vulkan, RDNA2 GPU path)

Model + C++ decode module (`cpp/`) + build/runbook: [README_ML2_Vulkan.md](README_ML2_Vulkan.md),
[docs/ML2_ONDEVICE_RUNBOOK.md](docs/ML2_ONDEVICE_RUNBOOK.md). Host-4090 Vulkan verified
(parity PASS); on-device: CPU ~15 FPS measured (yolo26n@640), Vulkan-GPU pending. ONNX exports (FP32/FP16/INT8) also provided.

## Datasets

- **DUT-Anti-UAV**: <https://github.com/wangdongdut/DUT-Anti-UAV>
- **Maciullo DroneDetectionDataset**: <https://github.com/Maciullo/DroneDetectionDataset> (HF mirror: `pathikg/drone-detection-dataset`)
- Leakage-safe merge (val = DUT official val only; both original test sets held out):
  `scripts/fetch_maciullo.py` → `merge_datasets.py` → `analyze_merge.py`.

## License

Datasets keep their own licenses; not redistributed here.
