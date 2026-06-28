# DroneAR — Magic Leap 2용 YOLO26 드론 탐지

> 🌐 English version: [README_English.md](README_English.md)

**DUT-Anti-UAV**로 **YOLO26** 드론(UAV) 탐지 모델 학습 → **Magic Leap 2(ML2)** 배포용 export.
재현 가능한 end-to-end 파이프라인이다.

- **학습 환경:** RTX 4090 24GB / Linux / CUDA (학습 전용)
- **추론 타깃:** ML2 — AMD "Mero" SoC (Zen2 쿼드코어 x86-64 CPU + RDNA2 iGPU), 16GB,
  AOSP Android 10 (API 29). **NVIDIA 아님** → 디바이스 TensorRT/CUDA 불가.
  검증 경로: **ONNX → ONNX Runtime(+MLSDK C API), CPU 백엔드 XNNPACK.**
- **모델 결정:** `yolo26n`(nano) 우선, **NMS-free one-to-one head 유지**, `imgsz=640`,
  INT8/FP16 export. CPU 추론으로 RDNA2 GPU는 120Hz AR 스테레오 렌더링에 양보.

> 상태: 전체 파이프라인 완료 (데이터 → 학습 → 평가 → ML2 export → 벤치 → Docker 검증).

---

## 성능 지표 (모델 선택 기준)

### 정확도 (150 epochs) — `weights/metrics.json` (test 기준; val 지표는 json 참조)

| 모델 | imgsz | test mAP50 | test mAP50-95 | Precision | Recall | Params(M) | FLOPs(G)@640 | best.pt |
|------|------:|----------:|-------------:|----------:|-------:|---------:|------------:|--------:|
| yolo26n | 640 | 0.951 | 0.648 | 0.963 | 0.922 | 2.4 | 5.2 | 5.4 MB |
| **yolo26n** | **960** | **0.968** | **0.699** | 0.976 | 0.936 | 2.4 | 5.2 | 5.5 MB |
| yolo26s | 640 | 0.958 | 0.681 | 0.968 | 0.945 | 9.5 | 20.5 | 20.3 MB |
| **yolo26s** | **960** | **0.970** | **0.723** | 0.981 | 0.956 | 9.5 | 20.5 | 20.4 MB |

> Params·FLOPs는 하드웨어 독립 복잡도다. **FLOPs(G)@640**: ultralytics fused 기준, **2×MAC 관례**
> (곱·합 각 1회 = MACs×2), 정밀도 무관 모델당 1값. @640 아키텍처 복잡도이며 imgsz 960의 실제
> 연산량은 약 2.25배다.

- imgsz **960이 640 대비 test mAP50-95 +4~5%p** (소형 객체 ~77% → 해상도 효과 큼). 단 추론 비용 ↑(입력 2.25배).
- 선택 가이드: 지연 우선 **yolo26n 640**, 정확도 우선 **960**. yolo26s는 정확도 상한선(파라미터 약 4배).
- 예측 예시(작은 드론, conf 0.78): `docs/demo/`.

### 추론 속도 — GPU (RTX 4090)

config: imgsz=640, batch=1(single-stream), warmup=30, iters=200, **순수 forward(전·후처리·NMS 제외)**,
torch CUDA(`cuda.Event` 계측), FPS = 1000/mean. 측정 하드웨어 **NVIDIA RTX 4090**. 원시 `weights/latency_gpu.md`.

| 모델 | 정밀도 | latency mean±std (ms) | FPS |
|------|--------|---------------------:|----:|
| yolo26n | FP32 | 2.40 ± 0.10 | 417 |
| yolo26n | FP16 | 2.48 ± 0.10 | 403 |
| yolo26s | FP32 | 2.44 ± 0.14 | 410 |
| yolo26s | FP16 | 2.57 ± 0.08 | 389 |

- INT8(GPU): TensorRT 엔진 빌드 시에만 측정(리포 INT8 ONNX는 CPU/XNNPACK용 QDQ Conv-only라 CUDA EP 비대표).
  미빌드 → **TODO**: `yolo export model=weights/yolo26n_drone_640.pt format=engine half=True device=0` 후 동일 프로토콜 측정.
- batch=1·작은 모델은 RTX 4090을 포화시키지 못해 런치/메모리 바운드 → 모델·정밀도 간 차이가 작다.

### 추론 속도 — CPU (i9-13900K, ONNX Runtime)

config: ORT **CPUExecutionProvider**, imgsz=640, batch=1, warmup=30, iters=200,
`intra_op_num_threads`=1·4 (inter_op=1, sequential), FPS = 1000/mean. 측정 하드웨어
**Intel i9-13900K**. 원시 `weights/latency_report.md`.

| 모델 | 정밀도 | 크기(MB) | t=1 ms | t=4 ms | t=1 FPS | t=4 FPS |
|------|--------|--------:|-------:|-------:|--------:|--------:|
| yolo26n | FP32 | 9.80 | 44.0 ± 0.5 | 13.2 ± 0.2 | 23 | 76 |
| yolo26n | FP16 | 4.97 | 45.5 ± 0.8 | 13.9 ± 0.2 | 22 | 72 |
| yolo26n | INT8 | 3.01 | **33.7 ± 0.9** | 15.1 ± 0.4 | **30** | 66 |
| yolo26s | FP32 | 38.17 | 149.6 ± 1.4 | 41.3 ± 0.9 | 7 | 24 |
| yolo26s | FP16 | 19.15 | 151.7 ± 1.5 | 42.4 ± 0.6 | 7 | 24 |
| yolo26s | INT8 | 10.24 | **86.6 ± 2.0** | 34.6 ± 0.7 | **12** | 29 |

- **FP16**: ORT CPU에 native fp16 커널 없음 → 속도 이득 없음(크기/이식성 옵션).
- **INT8**: 단일 스레드에서 가장 빠름. Conv-only QDQ라 4스레드에선 dequant 오버헤드로 이점 축소.
- 속도는 imgsz 640 기준. 960은 미측정(입력 2.25배).

### Export 산출물 (정밀도·크기) — NMS-free head, 출력 `[1,300,6]`

| 정밀도 | 파일 | 크기 | 비고 |
|--------|------|-----:|------|
| FP32 | `weights/yolo26n_drone_640_fp32.onnx` | 9.80 MB | 기준; opset17, static, simplified |
| FP16 | `weights/yolo26n_drone_640_fp16.onnx` | 4.97 MB | native `half=True`; float16 I/O |
| INT8 | `weights/yolo26n_drone_640_int8.onnx` | **3.01 MB** | static PTQ(QDQ), Conv-only, 200장 캘리브 |

**INT8 vs FP32** (동일 val 20장, conf 0.25): yolo26n 탐지 27→27(평균 IoU 0.961, |Δscore| 0.075),
yolo26s 27→26(평균 IoU 0.966, |Δscore| 0.103) → 저하 미미.

비교군/해상도 산출물: yolo26s_640 FP32 38.2 / FP16 19.2 / INT8 10.2 MB ·
imgsz 960(입력 `[1,3,960,960]`) yolo26n_960 10.0/5.1/**3.2** MB · yolo26s_960 38.4/19.3/10.5 MB
(`weights/yolo26{n,s}_drone_960_{fp32,fp16,int8}.onnx`).

---

## 모델 상세 (I/O)

ONNX를 추론 엔진에 통합할 때 필요한 입출력 방식에 대해서 간략히 설명한다 (imgsz 640 모델 기준; 960 변형은 입력·좌표가 960).

| 항목 | 사양 |
|------|------|
| 입력 | `images` `(1,3,640,640)` — float32(FP32·INT8) / float16(FP16) |
| 전처리 | **letterbox 640 · RGB · `/255` · CHW** (종횡비 보존 패딩, pad=114) |
| 출력 | `output0` `(1,300,6)` = `[x1,y1,x2,y2,score,class]`, 640 letterbox **픽셀** 좌표 |
| 후처리 | **NMS 불필요**(one-to-one head). `score ≥ 0.25` 필터 → letterbox 역산(패딩 빼고 scale로 나눔) → 원본 좌표 |
| 클래스 | `0 = drone` (단일 클래스, `nc=1`) |

INT8 모델도 입력은 float32다(Q/DQ는 그래프 내부 처리). 권장 conf 임계값 0.25는 디바이스에서 튜닝한다.

---

## 리포지토리 구조

```
scripts/   voc2yolo.py  dataset_stats.py  train.py  train_all.sh
           eval.py  predict.py  export.py  bench_latency.py  bench_gpu.py
configs/   dut_drone.yaml
weights/   yolo26{n,s}_drone_{640,960}.pt
           yolo26{n,s}_drone_{640,960}_{fp32,fp16,int8}.onnx
           metrics.json  latency_report.md(CPU)  latency_gpu.md(GPU)
docs/demo/ 예측 예시 이미지
Dockerfile · docker-compose.yml · .dockerignore · requirements.txt · README.md
```

---

## 데이터셋

DUT-Anti-UAV는 수동 준비. 아래 PASCAL VOC 구조로 `/mnt/ssd_0/dataset/DUT`에 배치/압축해제한다.
변환 스크립트는 이 트리를 **수정하지 않는다**(read-only).

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

- 단일 클래스: 원본 `UAV`(10,109개) → `0: drone`(`nc=1`) 매핑.
- 객체 없는 train 3장 → 빈 `.txt`(negative). 불량 박스(w≤0/h≤0) 1개 스킵.

**변환 (원본 read-only):**
```bash
python scripts/voc2yolo.py        # --src /mnt/ssd_0/dataset/DUT  --dst /mnt/ssd_0/dataset/dut_yolo
python scripts/dataset_stats.py   # 박스 크기 히스토그램 + 샘플 박스 시각화 -> dut_yolo/_viz/
```

**박스 크기 분포 — 소형 객체 위주** (imgsz/P2 결정 근거).
정규화 변 `sqrt(w·h)`: 중앙값 **0.0226**(~14.5px @640), p25 0.0163, p75 0.0451, max 0.84.

| 크기 구간 (@imgsz 640) | 비율 |
|---|---:|
| SMALL (변 <32px) | **76.6%** |
| MEDIUM (32–96px) | 13.1% |
| LARGE (변 >96px) | 10.3% |
| tiny (<13px, 정규화변 <0.02) | 40.6% |

→ 드론 대부분 소형. 기본 `imgsz=640`(ML2 타깃) 유지. 소형 recall 향상 레버는 **imgsz=960·P2 head**.

---

## 환경 구성

### 방법 A — Docker (권장, 협업자 재현용)

Docker Hub 이미지: **`hanmyeongil/yolo26:v1`** (빌드 없이 바로 사용).

```bash
docker compose pull      # Docker Hub에서 이미지 받기 (또는 docker compose build 로 직접 빌드)
docker compose run --rm dronear python scripts/voc2yolo.py
docker compose run --rm dronear python scripts/train.py
docker compose run --rm dronear python scripts/export.py
```

> ⚠️ **작업 경로 필수 설정.** `docker compose`는 **`docker-compose.yml`이 있는 repo 루트에서**
> 실행한다. 다른 경로에서 실행하면 compose 파일·상대 볼륨(`./scripts`, `./weights`, `./runs`)을
> 못 찾아 엉뚱한(새) 경로 기준으로 동작한다. 컨테이너 작업 디렉터리는 `working_dir=/workspace`
> 고정이며, `scripts/`·`configs/`·`weights/`·`runs/`가 여기에 마운트된다.
>
> `docker run`을 직접 쓸 때도 `-w /workspace` + repo 루트를 `/workspace`로 마운트해야 한다:
> ```bash
> docker run --rm --gpus all \
>   -v "$PWD":/workspace -w /workspace \
>   -v /mnt/ssd_0/dataset:/mnt/ssd_0/dataset \
>   hanmyeongil/yolo26:v1 python scripts/export.py
> ```

데이터셋은 호스트 경로 → 동일 컨테이너 경로로 마운트 → `configs/dut_drone.yaml`이 네이티브/컨테이너
양쪽 동작. **다른 머신은 `docker-compose.yml`의 데이터셋 볼륨 + config `path:` 한 줄을 자기
데이터 경로로 변경**한다(안 하면 컨테이너가 데이터를 못 찾음).

**재현성 검증 완료.** 베이스 `ultralytics/ultralytics:latest` + `onnxruntime`/`onnxslim`/
`onnxconverter-common`, 기본 polars → `polars-lts-cpu` 교체 → 동작 GPU 이미지(컨테이너 내 CUDA OK).
컨테이너 안에서 `scripts/export.py` 실행 → 호스트 venv와 동일 산출물(FP32 9.80MB, FP16 4.97MB
native-half, INT8 3.01MB), 모두 ORT 로드·출력 `[1,300,6]` 확인.

### 방법 B — venv (빠른 개발 루프)

```bash
python3 -m venv .venv && . .venv/bin/activate
# torch는 호스트 CUDA 12.8 드라이버에 맞는 cu128 빌드 먼저 (아래 Troubleshooting 참고)
pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python scripts/voc2yolo.py
python scripts/train.py
```

---

## 재현 절차 (전체 명령)

각 단계는 Docker·venv 형태 모두 제공.

| 단계 | Docker | venv |
|------|--------|------|
| VOC→YOLO 변환 | `docker compose run --rm dronear python scripts/voc2yolo.py` | `python scripts/voc2yolo.py` |
| 데이터 통계 | `... python scripts/dataset_stats.py` | `python scripts/dataset_stats.py` |
| 학습(단일) | `... python scripts/train.py --model yolo26n.pt --name yolo26n_drone_640` | `python scripts/train.py ...` |
| 학습(n+s, 150ep) | `... bash scripts/train_all.sh` | `bash scripts/train_all.sh` |
| 평가(val+test) | `... python scripts/eval.py --weights weights/yolo26n_drone_640.pt` | `python scripts/eval.py ...` |
| Export ONNX/FP16/INT8 | `... python scripts/export.py --weights weights/yolo26n_drone_640.pt --stem yolo26n_drone_640` | `python scripts/export.py ...` |
| 속도 벤치 GPU(4090) | `... python scripts/bench_gpu.py` | `python scripts/bench_gpu.py` |
| 속도 벤치 CPU(ORT) | `... python scripts/bench_latency.py --stems yolo26n_drone_640 yolo26s_drone_640` | `python scripts/bench_latency.py ...` |
| 예측 데모 | `... python scripts/predict.py --weights weights/yolo26n_drone_640.pt` | `python scripts/predict.py ...` |

**학습 설정(ML2 baseline):** `yolo26n.pt`, `imgsz=640`, `epochs=150`, `patience=40`,
`batch=-1`(자동 → 4090에서 ~35), `cache=disk`, NMS-free head 유지. `yolo26s`는 정확도 비교군.
5-epoch 스모크 수렴 확인(mAP50 0.62→0.81).

### Troubleshooting (환경 이슈 — requirements 반영)

| 증상 | 원인 | 해결 |
|---|---|---|
| `cuda.is_available()=False`, "driver too old" | ultralytics가 torch `cu130` 끌어옴; 호스트는 CUDA 12.8 | `torch==2.11.0+cu128`(최신 cu128) 설치 |
| **Bus error(SIGBUS)** — 첫 체크포인트 저장 시 | `polars` 1.42 휠 import SIGBUS; ultralytics가 매 epoch `results.csv`를 polars로 읽음 | **`polars-lts-cpu`** 교체 |
| `cache=ram` SIGBUS | DataLoader가 캐시 배열을 `/dev/shm` 공유 | `cache=disk`(기본) 또는 `--cache False` |
| TFLite export 실패 (`tf.tile_36` rank 에러) | onnx2tf 1.28.8이 YOLO26 NMS-free head `Tile` 미지원 | ONNX 경로 사용; 필요시 onnx2tf 버전/`param_replacement.json` |

---

## 개선 옵션 (소형 객체)

데이터셋 ~77% 소형 → 소형 recall 향상 레버:
- **imgsz 960 재학습:** `python scripts/train.py --imgsz 960 --name yolo26n_drone_960`
- **P2 head**(stride-4 세밀 특징): `--model yolo26-p2.yaml` (from scratch) 후 재 export
- 정확도 여유 필요 시 **yolo26s**

> 모델 추론·앱 구현·디바이스 배포는 본 리포 범위 밖이다. 협업자에게 **가중치·ONNX 산출물**
> (`weights/`)을 전달한다.

---

## 라이선스 / 비고

데이터셋(DUT-Anti-UAV)은 자체 라이선스를 따른다. 여기서 재배포하지 않는다.
