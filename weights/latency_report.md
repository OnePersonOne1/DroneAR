# Latency report — yolo26n_drone_640

> **Directional estimate only.** ONNX Runtime / CPUExecutionProvider on an
> x86-64 desktop CPU. This is a proxy for the Magic Leap 2 Zen2 mobile CPU,
> not a measurement of it. Final numbers require on-device ML2 ADB profiling.

- imgsz=640, runs=100 (warmup=10), batch=1
- threads tested: [1, 4] (4 ≈ ML2 Zen2 quad-core)

| Precision | File | Size (MB) | t=1 mean±std (ms) | t=4 mean±std (ms) |
|---|---|---:|---:|---:|
| FP32 | `yolo26n_drone_640_fp32.onnx` | 9.80 | 41.88 ± 1.45 | 12.86 ± 0.49 |
| FP16 | `yolo26n_drone_640_fp16.onnx` | 4.97 | 42.89 ± 0.89 | 13.43 ± 0.29 |
| INT8 | `yolo26n_drone_640_int8.onnx` | 3.01 | 30.23 ± 0.90 | 14.11 ± 0.38 |

## FP32 vs INT8 accuracy (same images, conf=0.25)

- images: 20
- detections: FP32 **27** vs INT8 **27**
- matched (IoU≥0.5): 27
- mean |Δscore| on matches: 0.0749
- mean IoU on matches: 0.9607

## Notes
- **FP16** has no CPU speedup (ORT CPU has no native fp16 kernels — values are
  up/down-cast); it is a size/portability option, not a CPU latency win.
- **INT8** is the smallest model and fastest single-thread. At 4 threads the
  Conv-only QDQ dequant overhead narrows the gap on this x86 desktop; the ML2
  target uses ONNX Runtime XNNPACK, which handles INT8 differently — another
  reason these numbers are directional and need on-device confirmation.
- INT8 quantizes Conv layers only; the NMS-free detection head stays float so
  the small score values are not crushed to zero.
