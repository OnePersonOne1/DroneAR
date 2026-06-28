# CPU latency — 13th Gen Intel(R) Core(TM) i9-13900K

데스크톱 CPU 실측(추정/기기 비교 없음).

- ONNX Runtime CPUExecutionProvider, imgsz=640, batch=1 (single-stream)
- warmup=30, iters=200, intra_op_num_threads=[1, 4] (inter_op=1, sequential)
- FPS = 1000 / mean_latency_ms

| 모델 | 정밀도 | 크기(MB) | threads=1 mean±std(ms) | threads=4 mean±std(ms) | threads=1 FPS | threads=4 FPS |
|---|---|---:|---:|---:|---:|---:|
| yolo26n_drone_640 | FP32 | 9.80 | 43.99 ± 0.49 | 13.19 ± 0.20 | 23 | 76 |
| yolo26n_drone_640 | FP16 | 4.97 | 45.47 ± 0.82 | 13.86 ± 0.24 | 22 | 72 |
| yolo26n_drone_640 | INT8 | 3.01 | 33.72 ± 0.89 | 15.14 ± 0.36 | 30 | 66 |
| yolo26s_drone_640 | FP32 | 38.17 | 149.59 ± 1.44 | 41.28 ± 0.94 | 7 | 24 |
| yolo26s_drone_640 | FP16 | 19.15 | 151.73 ± 1.47 | 42.42 ± 0.56 | 7 | 24 |
| yolo26s_drone_640 | INT8 | 10.24 | 86.63 ± 2.04 | 34.56 ± 0.70 | 12 | 29 |

## FP32 vs INT8 정확도 — yolo26n_drone_640 (동일 이미지 20장, conf=0.25)

- 탐지: FP32 **27** vs INT8 **27**, 매칭(IoU≥0.5) 27
- 평균 |Δscore| 0.0749, 평균 IoU 0.9607

## FP32 vs INT8 정확도 — yolo26s_drone_640 (동일 이미지 20장, conf=0.25)

- 탐지: FP32 **27** vs INT8 **26**, 매칭(IoU≥0.5) 27
- 평균 |Δscore| 0.1026, 평균 IoU 0.9658

## Notes
- **FP16**: ORT CPU에 native fp16 커널이 없어 up/down-cast → CPU 속도 이득 없음(크기/이식성 옵션).
- **INT8**: Conv 레이어만 양자화(QDQ). NMS-free detection head는 float 유지 → 작은 score가 0으로 뭉개지지 않음.
