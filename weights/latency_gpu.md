# GPU latency — NVIDIA GeForce RTX 4090

- imgsz=640, batch=1 (single-stream), warmup=30, iters=200
- 순수 forward(전·후처리·NMS 제외). NMS-free one-to-one head.
- device=cuda:0, torch CUDA. FPS = 1000 / mean_latency_ms.

| 모델 | 정밀도 | latency mean±std (ms) | FPS |
|---|---|---:|---:|
| yolo26n_drone_640 | FP32 | 2.401 ± 0.096 | 416.5 |
| yolo26n_drone_640 | FP16 | 2.484 ± 0.100 | 402.7 |
| yolo26s_drone_640 | FP32 | 2.441 ± 0.139 | 409.7 |
| yolo26s_drone_640 | FP16 | 2.571 ± 0.081 | 388.9 |

INT8(GPU): TensorRT 엔진 빌드 시에만 측정. 미빌드 → TODO. 리포의 INT8 ONNX는 CPU/XNNPACK용 QDQ Conv-only라 CUDA EP에서 비대표.
