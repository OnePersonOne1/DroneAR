# DroneAR — YOLO26 / D-FINE Drone Detection for Magic Leap 2

> One-page English summary. Full documentation (Korean): [README.md](README.md)

Train drone (UAV) detectors — **YOLO26** and **D-FINE** — on **DUT-Anti-UAV** (+ **Maciullo
DroneDetectionDataset** merged, 10× training data) → deploy to **Magic Leap 2** (on-device)
and/or a cloud GPU.

- **Training**: RTX 4090 / Linux / CUDA. Reproduce: `docker compose pull` (image `hanmyeongil/yolo26:v1`) or `pip install -r requirements.txt`.
- **Inference targets**: ML2 (AMD Zen2 x86-64 + RDNA2 iGPU, Android 10 / API 29 — no NVIDIA) via ONNX Runtime CPU or ncnn-Vulkan; cloud RTX 4090 via PyTorch.

## Weights download

`weights/` is split by family: **`weights/yolo26/`** (YOLO) and **`weights/d_fine/`** (D-FINE).
Files **<100MB are included in the repo**; only **D-FINE-L (477MB) exceeds GitHub's 100MB limit → Google Drive only**.

| model | imgsz | release file | size | location / download |
|---|--:|---|--:|---|
| **D-FINE-L** | 960 | `dfine_l_drone_960_mergedataset_120epoch.pth` | 477MB | [⬇ Google Drive](https://drive.google.com/file/d/17xsaKm4ziOSl03LQ-OZxgvVmkIRsd6Qd/view?usp=sharing) (Drive only) |
| D-FINE-N | 640 | `dfine_n_drone_640_mergedataset_220epoch.pth` | 58MB | repo `weights/d_fine/` |
| yolo26l-P2 | 960 | `yolo26l_drone_960p2_mergedataset_100epoch.pt` | 50MB | repo `weights/yolo26/` |
| yolo26n-P2 | 960 | `yolo26n_drone_960p2_mergedataset_100epoch.pt` | 5.9MB | repo `weights/yolo26/` |
| yolo26n merged | 640 | `yolo26n_drone_640_mergedataset_{100,300}epoch.pt` (+onnx/ncnn) | ~5MB | repo `weights/yolo26/` |
| yolo26{n,s} (old) | 640/960 | `yolo26{n,s}_drone_{640,960}.pt` (+onnx) | 5–20MB | repo `weights/yolo26/` |

> After downloading, place D-FINE-L at `weights/d_fine/dfine_l_drone_960_mergedataset_120epoch.pth` to match the eval/inference paths.

## Models & results (unified evaluation, held-out test sets)

All models measured with the same protocol: held-out test (DUT-test 2200 / Maciullo-test 2625) ·
faster-coco-eval · conf 0.25 · IoU-match 0.5. far = recall of objects <16px (side @640).
Full table: [reports/master_table.md](reports/master_table.md).

| model | train | imgsz | DUT AP50 / AP50-95 / far | Maci AP50 / AP50-95 / far | use |
|---|---|--:|---|---|---|
| yolo26n | DUT | 640 | 0.923 / 0.630 / 0.925 | 0.523 / 0.196 / 0.359 | legacy baseline |
| yolo26n (300ep) | merged | 640 | 0.915 / 0.632 / 0.876 | 0.791 / 0.403 / 0.748 | **on-device (ML2)** |
| yolo26n-P2 | merged | 960 | 0.924 / 0.673 / 0.933 | 0.831 / 0.429 / 0.793 | cloud, lightweight |
| yolo26l-P2 | merged | 960 | 0.959 / 0.755 / 0.960 | 0.842 / 0.437 / 0.783 | cloud YOLO (infer at 1280) |
| D-FINE-N | merged | 640 | 0.950 / 0.706 / 0.947 | 0.865 / 0.423 / 0.838 | DETR, CPU-efficient |
| **D-FINE-L** | merged | 960 | **0.973 / 0.778 / 0.987** | **0.907 / 0.456 / 0.798** | **cloud flagship** |

- **D-FINE-L is best across the board** (DUT AP50 0.973, Maci AP50 0.907, DUT far 0.987) — cloud-only (not portable to ncnn-Vulkan due to `grid_sample`).
- DUT-only models collapse on Maciullo (AP50 ~0.5) → domain merge is required for transfer.
- Inference resolution: YOLO gains from upscaling (640→1280 far 0.876→0.944); **D-FINE collapses off its training scale** → keep D-FINE inference at the training size, square resize.

**Unified latency** (pure torch forward, batch 1, fp32): GPU RTX 4090 / CPU Ryzen 9 7950X 8-thread —
yolo26n@640 **340 / 33 FPS**, D-FINE-N@640 195 / 24 FPS, yolo26l-P2@960 113 / 1.6 FPS, D-FINE-L@960 84 / 2.1 FPS.

- Ablation (per-factor attribution): [reports/ablation_matrix.md](reports/ablation_matrix.md)
- Far/small-object analysis & resolution sweep: [reports/far_drone_p2_960.md](reports/far_drone_p2_960.md)
- YOLO26 n–x FPS on 4090: [reports/yolo26_family_fps_4090.md](reports/yolo26_family_fps_4090.md)

## Deployment recommendation

| path | model | why |
|---|---|---|
| Cloud (4090), best accuracy | **D-FINE-L@960** | best on every metric; cloud-only |
| Cloud, best YOLO | yolo26l-P2@960, infer at 1280 | DUT far 0.968, FP16 103 FPS |
| On-device (ML2) | **yolo26n merged-300ep @640** | lowest FP/img; CPU ~15 FPS measured; ncnn-Vulkan portable |

## ML2 deployment (ncnn-Vulkan, RDNA2 GPU path)

Model + C++ decode module (`cpp/`) + build/runbook: [README_ML2_Vulkan.md](README_ML2_Vulkan.md),
[docs/ML2_ONDEVICE_RUNBOOK.md](docs/ML2_ONDEVICE_RUNBOOK.md). Host-4090 Vulkan verified
(parity PASS); on-device: CPU ~15 FPS measured (yolo26n@640), Vulkan-GPU pending.
ONNX exports (FP32/FP16/INT8, NMS-free head, output `[1,300,6]`) also provided. D-FINE is
CPU-only on ML2 (`grid_sample` unsupported by ncnn-Vulkan) → on-device mainline stays yolo26n.

## Datasets

- **DUT-Anti-UAV**: <https://github.com/wangdongdut/DUT-Anti-UAV>
- **Maciullo DroneDetectionDataset**: <https://github.com/Maciullo/DroneDetectionDataset> (HF mirror: `pathikg/drone-detection-dataset`)
- Leakage-safe merge (val = DUT official val only; both original test sets held out):
  `scripts/fetch_maciullo.py` → `merge_datasets.py` → `analyze_merge.py`.

## License

Datasets keep their own licenses; not redistributed here.
