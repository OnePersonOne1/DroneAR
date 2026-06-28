# DroneAR — Magic Leap 2용 YOLO26 드론 탐지

> 🌐 English version: [README_English.md](README_English.md)

**DUT-Anti-UAV** 데이터셋으로 **YOLO26** 드론(UAV) 객체 탐지 모델을 학습하고, **Magic Leap 2 (ML2)**
에 배포 가능한 형태로 export 하는 **재현 가능한 end-to-end 파이프라인**입니다.

- **학습 환경:** RTX 4090 24GB / Linux / CUDA (학습 전용)
- **추론 타깃:** Magic Leap 2 — AMD "Mero" SoC (Zen2 쿼드코어 x86-64 CPU + RDNA2 iGPU),
  16GB, AOSP Android 10 (API 29). **NVIDIA 아님** → 디바이스에서 TensorRT/CUDA 불가.
  검증된 on-device 경로: **ONNX → ONNX Runtime (+MLSDK C API), CPU 백엔드 XNNPACK.**
- **모델 결정:** `yolo26n`(nano) 우선, **NMS-free one-to-one head 유지**, `imgsz=640`,
  INT8 / FP16 export. CPU 추론으로 RDNA2 GPU를 비워 120Hz AR 스테레오 렌더링에 양보.

> 상태: 전체 파이프라인 완료 (데이터 → 학습 → 평가 → ML2 export → 벤치 → Docker 검증).

---

## 리포지토리 구조

```
scripts/   voc2yolo.py  dataset_stats.py  train.py  train_all.sh
           eval.py  predict.py  export.py  bench_latency.py
configs/   dut_drone.yaml
weights/   yolo26{n,s}_drone_640.pt  yolo26n_drone_640_{fp32,fp16,int8}.onnx
           metrics.json  latency_report.md
docs/demo/ 예측 예시 이미지
Dockerfile · docker-compose.yml · .dockerignore · requirements.txt · README.md
```

---

## 데이터셋

DUT-Anti-UAV는 수동으로 아래 PASCAL VOC 구조로 `/mnt/ssd_0/dataset/DUT`에
배치/압축해제 해야 한다. (변환 스크립트는 이 트리를 **수정하지 않는다.**):

```
/mnt/ssd_0/dataset/DUT/{train,val,test}/{img,xml}
  img/  *.jpg
  xml/  *.xml   (VOC: <size>, <object><name>, <bndbox> xmin/ymin/xmax/ymax)
```

| Split | 이미지 | 라벨 | 박스 | Negative | Skip(불량박스) |
|-------|-------:|----:|----:|---------:|--------------:|
| train | 5200 | 5200 | 5243 | 3 | 0 |
| val   | 2600 | 2600 | 2620 | 0 | 1 |
| test  | 2200 | 2200 | 2245 | 0 | 0 |
| **합계** | **10000** | **10000** | **10108** | **3** | **1** |

- 단일 클래스: 원본 라벨 `UAV`(객체 10,109개) → 클래스 `0: drone`(`nc=1`)으로 매핑.
- train 이미지 3장은 객체 없음 → 빈 `.txt`(negative)로 생성. 불량 박스(w≤0/h≤0) 1개 스킵.

**변환 (원본 read-only):**
```bash
python scripts/voc2yolo.py        # --src /mnt/ssd_0/dataset/DUT  --dst /mnt/ssd_0/dataset/dut_yolo
python scripts/dataset_stats.py   # 박스 크기 히스토그램 + 샘플 박스 시각화 -> dut_yolo/_viz/
```

**박스 크기 분포 — 소형 객체 위주** (imgsz/P2 결정 근거):
정규화 변 길이 `sqrt(w·h)`: 중앙값 **0.0226** (~14.5px @640), p25 0.0163, p75 0.0451, max 0.84.

| 크기 구간 (@imgsz 640) | 비율 |
|---|---:|
| SMALL (변 <32px) | **76.6%** |
| MEDIUM (32–96px) | 13.1% |
| LARGE (변 >96px) | 10.3% |
| tiny (<13px, 정규화변 <0.02) | 40.6% |

→ 드론 대부분이 작음/매우 작음. 기본은 `imgsz=640`(ML2 타깃) 유지하되, 소형 객체 recall 향상을 위해
**imgsz=960 과 P2 head를 1차 정확도 레버로 권장**한다.

---

## 환경 구성

### 방법 A — Docker (권장, 협업자 재현용)

```bash
docker compose build
docker compose run --rm dronear python scripts/voc2yolo.py
docker compose run --rm dronear python scripts/train.py
docker compose run --rm dronear python scripts/export.py
```

데이터셋을 동일한 컨테이너 경로로 마운트하므로 `configs/dut_drone.yaml`이
네이티브/컨테이너 양쪽에서 그대로 동작한다. 다른 머신에서는 `docker-compose.yml`의 데이터셋
볼륨과 config의 `path:` 한 줄만 바꾸면 된다.

**재현성 검증 완료:** `docker compose build` (베이스 `ultralytics/ultralytics:latest` +
`onnxruntime`/`onnxslim`/`onnxconverter-common`, 기본 polars를 `polars-lts-cpu`로 교체)로
동작하는 GPU 이미지 생성(컨테이너 내 CUDA 접근 가능). 컨테이너 안에서
`docker compose run --rm dronear python scripts/export.py --weights weights/yolo26n_drone_640.pt
--stem yolo26n_drone_640 --outdir weights/docker_verify`를 실행해 호스트 venv와 동일한
산출물(FP32 9.80 MB, FP16 4.97 MB native-half, INT8 3.01 MB)을 생성, 모두 ORT에서 출력
`[1,300,6]`으로 로드됨을 확인했다.

### 방법 B — venv (빠른 개발 루프)

```bash
python3 -m venv .venv && . .venv/bin/activate
# torch는 이 호스트의 CUDA 12.8 드라이버에 맞는 cu128 빌드를 먼저 설치 (아래 Troubleshooting 참고)
pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python scripts/voc2yolo.py
python scripts/train.py
```

---

## 재현 절차 (전체 명령)

각 단계는 Docker 형태와 venv 형태를 모두 제공합니다.

| 단계 | Docker | venv |
|------|--------|------|
| VOC→YOLO 변환 | `docker compose run --rm dronear python scripts/voc2yolo.py` | `python scripts/voc2yolo.py` |
| 데이터 통계 | `... python scripts/dataset_stats.py` | `python scripts/dataset_stats.py` |
| 학습(단일) | `... python scripts/train.py --model yolo26n.pt --name yolo26n_drone_640` | `python scripts/train.py ...` |
| 학습(n+s, 150ep) | `... bash scripts/train_all.sh` | `bash scripts/train_all.sh` |
| 평가(val+test) | `... python scripts/eval.py --weights weights/yolo26n_drone_640.pt` | `python scripts/eval.py ...` |
| Export ONNX/FP16/INT8 | `... python scripts/export.py --weights weights/yolo26n_drone_640.pt --stem yolo26n_drone_640` | `python scripts/export.py ...` |
| 지연 벤치 | `... python scripts/bench_latency.py --stem yolo26n_drone_640` | `python scripts/bench_latency.py ...` |
| 예측 데모 | `... python scripts/predict.py --weights weights/yolo26n_drone_640.pt` | `python scripts/predict.py ...` |

**학습 설정(ML2 baseline):** `yolo26n.pt`, `imgsz=640`, `epochs=150`, `patience=40`,
`batch=-1`(자동 → 4090에서 ~35), `cache=disk`, NMS-free one-to-one head 유지. `yolo26s`는
정확도 비교군. 5-epoch 스모크로 수렴 확인(5 epoch 만에 mAP50 0.62→0.81).

### Troubleshooting (환경 이슈 — requirements에 반영됨)

| 증상 | 원인 | 해결 |
|---|---|---|
| `cuda.is_available()=False`, "driver too old" | ultralytics가 torch `cu130`을 끌어옴; 이 호스트 드라이버는 CUDA 12.8 | `torch==2.11.0+cu128`(최신 cu128 빌드) 설치 |
| **Bus error (SIGBUS)** — 첫 체크포인트 저장 시 | `polars` 1.42 휠이 이 CPU에서 import 시 SIGBUS; ultralytics가 매 epoch `results.csv`를 polars로 읽음 | **`polars-lts-cpu`** 로 교체 |
| `cache=ram` 에서 SIGBUS | DataLoader가 캐시 배열을 `/dev/shm`로 공유 | `cache=disk`(기본) 또는 `--cache False` 사용 |
| TFLite export 실패 (`tf.tile_36` rank 에러) | onnx2tf 1.28.8이 YOLO26 NMS-free head의 `Tile` 변환 미지원 | ONNX 경로 사용(주 경로); 필요시 onnx2tf 버전 변경/`param_replacement.json` |

---

## 결과

### 정확도 (150 epochs, imgsz 640) — `weights/metrics.json` 기준

| 모델 | Split | mAP50 | mAP50-95 | Precision | Recall | 파라미터 | best.pt |
|------|-------|------:|---------:|----------:|-------:|--------:|--------:|
| **yolo26n** (ML2 메인) | val | 0.911 | 0.583 | 0.958 | 0.872 | 2.4M | 5.4 MB |
| yolo26n | test | **0.951** | 0.648 | 0.963 | 0.922 | | |
| yolo26s (비교군) | val | 0.929 | 0.617 | 0.963 | 0.903 | 9.5M | 20.3 MB |
| yolo26s | test | **0.958** | 0.681 | 0.968 | 0.945 | | |

yolo26s는 yolo26n 대비 test mAP50 약 +0.7%p / mAP50-95 약 +3%p 향상이나, 파라미터·GFLOPs는
약 4배(5.2→20.5)입니다. 따라서 ML2 CPU 타깃에는 **yolo26n을 배포 모델로 권장**하며, 여유가
있으면 yolo26s가 정확도 상한선이다. 예측 예시(매우 작은 드론, conf 0.78): `docs/demo/`.

### Export 정밀도 — yolo26n (ML2 메인), imgsz 640, NMS-free head, 출력 `[1,300,6]`

| 정밀도 | 파일 | 크기 | 비고 |
|--------|------|-----:|------|
| FP32 | `weights/yolo26n_drone_640_fp32.onnx` | 9.80 MB | 기준; opset17, static, simplified |
| FP16 | `weights/yolo26n_drone_640_fp16.onnx` | 4.97 MB | native `half=True`; float16 I/O |
| INT8 | `weights/yolo26n_drone_640_int8.onnx` | **3.01 MB** | static PTQ (QDQ), Conv-only, 200장 캘리브 |

**INT8 vs FP32 정확도** (동일 val 이미지 20장, conf 0.25): 탐지 **27 → 27**, 전부 IoU≥0.5로 매칭,
평균 IoU 0.961, 평균 |Δscore| 0.075 → 정확도 저하 미미.

비교군 **yolo26s**도 동일 경로로 export 했다: FP32 38.2 MB / FP16 19.2 MB / INT8 10.2 MB
(`weights/yolo26s_drone_640_{fp32,fp16,int8}.onnx`).

### Dev-CPU 지연 (방향성 추정치, **ML2 아님**) — `weights/latency_report.md`

> ⚠️ **x86-64 데스크톱 CPU**(i9-13900K)에서 ONNX Runtime / CPUExecutionProvider로 측정.
> ML2 Zen2 모바일 CPU의 *방향성* 추정치이며 실제 측정값이 **아니다**. 최종 수치는 ML2
> on-device ADB 프로파일링이 필요하다(디바이스는 ORT + XNNPACK 사용).

| 정밀도 | threads=1 (ms) | threads=4 (ms) | 크기 |
|--------|---------------:|---------------:|-----:|
| FP32 | 41.9 ± 1.5 | 12.9 ± 0.5 | 9.80 MB |
| FP16 | 42.9 ± 0.9 | 13.4 ± 0.3 | 4.97 MB |
| INT8 | **30.2 ± 0.9** | 14.1 ± 0.4 | **3.01 MB** |

INT8은 크기와 단일 스레드 지연에서 유리하다. 4 스레드에서는 Conv-only QDQ의 dequant 오버헤드로
x86에서 격차가 좁아진다(ML2의 XNNPACK은 동작이 다름). FP16은 CPU 속도 이득이 없으며(ORT CPU에
native fp16 커널 없음) 크기/이식성 옵션이다.

---

## Magic Leap 2 배포 (다음 단계 가이드)

**권장 산출물:** `weights/yolo26n_drone_640_int8.onnx` (3.0 MB) 또는 FP32 기준본
`..._fp32.onnx` (9.8 MB). 경로: **ONNX (opset 17, NMS-free) → ONNX Runtime (+ MLSDK C API),
CPU Execution Provider + XNNPACK.** CPU 추론으로 RDNA2 iGPU를 120Hz AR 스테레오 렌더링에 비워둔다.
(ML2는 AMD라 TensorRT/CUDA 불가.)

**출력 텐서**는 one-to-one head에서 나오는 `(1, 300, 6)` = `[x1, y1, x2, y2, score, class]`라
디바이스에서 **NMS 불필요** — `score` 임계값만 적용하면 됩니다. 좌표는 640×640 letterbox 입력
좌표계 기준이며, letterbox를 역산(패딩 빼고 스케일로 나눔)해 카메라 프레임 좌표로 매핑한다.

**On-device 앱 파이프라인:**
1. ML2 카메라 프레임 획득 (MLSDK 카메라/perception API 등).
2. 전처리: **640×640 letterbox, BGR→RGB, `/255`, HWC→CHW, float32** (INT8 모델도 동일 float
   입력 — Q/DQ는 내부 처리). *(스크립트는 종횡비 보존 위해 letterbox 사용; 단순 resize-640도
   되지만 작은 드론이 왜곡됨.)*
3. CPU EP(XNNPACK)로 ORT `Run`. 렌더링/perception용 코어 확보를 위해 `intra_op_num_threads ≈ 3`
   권장.
4. 300개 행 각각에서 `score ≥ 임계값` 유지(초기 ~0.25, on-device 튜닝).
5. letterbox 역산 → 박스를 원본 카메라 해상도로 스케일.
6. 박스 위치에 AR 오버레이 렌더(월드 앵커 quad 또는 HUD 마커).

**작은 드론이 미검출되면** (본 데이터셋은 약 77%가 소형 객체): **`imgsz=960`** 으로 재학습
(`python scripts/train.py --imgsz 960 --name yolo26n_drone_960`) 하거나, stride-4의 더 세밀한
특징을 위해 **P2 detection head** 추가(`--model yolo26-p2.yaml`, from scratch) 후 큰 해상도로
재 export. 지연 예산에 여유가 있으면 yolo26s가 정확도 상위 대안이다.

**한계 / 정직한 고지:** 본 리포의 지연 수치는 x86-64 데스크톱 CPU 측정값으로 ML2 Zen2 모바일
CPU의 **방향성 추정치**일 뿐 실측이 아닙니다. 실제 on-device 지연/정확도는 ML2 ADB 프로파일링으로
확인해야 한다. INT8은 크기·단일 스레드 지연에서 가장 유리하며, 멀티 스레드 이득은 디바이스 XNNPACK
커널에 따라 달라질 수 있다.

**선택적 대안 (현재 미생성):** TFLite INT8 (`format='tflite', int8=True`) — ML2의 TFLite +
NNAPI/XNNPACK 경로용. **시도했으나 실패**: 현재 `onnx2tf` 1.28.8이 YOLO26 NMS-free head의
`Tile` 연산을 변환하지 못함(`model.23/Tile` rank 불일치 → `Shape must be rank 3 but is rank 1`).
ultralytics `export(format='tflite')`·onnx2tf 직접 호출 모두 동일 한계(Python 3.13 + 최신
YOLO26 조합). ML2의 **1순위 검증 경로는 ONNX Runtime**이라 영향은 없으며, TFLite가 필요하면
onnx2tf 버전 변경 또는 `param_replacement.json`으로 해당 노드를 수동 보정해야 함. (TFLite 전용
툴체인은 기존 `.venv` 보호를 위해 별도 `.venv_tflite`에 격리.)

---

## 라이선스 / 비고

데이터셋(DUT-Anti-UAV)은 자체 라이선스를 따르며 여기서 재배포하지 않는다.
