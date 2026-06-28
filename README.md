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

### 정확도 (150 epochs, imgsz 640) — `weights/metrics.json`

| 모델 | Split | mAP50 | mAP50-95 | Precision | Recall | 파라미터 | best.pt |
|------|-------|------:|---------:|----------:|-------:|--------:|--------:|
| **yolo26n** (메인) | val | 0.911 | 0.583 | 0.958 | 0.872 | 2.4M | 5.4 MB |
| yolo26n | test | **0.951** | 0.648 | 0.963 | 0.922 | | |
| yolo26s (비교군) | val | 0.929 | 0.617 | 0.963 | 0.903 | 9.5M | 20.3 MB |
| yolo26s | test | **0.958** | 0.681 | 0.968 | 0.945 | | |

yolo26s: yolo26n 대비 test mAP50 +0.7%p / mAP50-95 +3%p, 단 파라미터·GFLOPs 약 4배(5.2→20.5).
CPU 추론 타깃 → **yolo26n 권장**, 여유 시 yolo26s가 정확도 상한선. 예측 예시(작은 드론, conf 0.78):
`docs/demo/`.

### Export 정밀도 — yolo26n, imgsz 640, NMS-free head, 출력 `[1,300,6]`

| 정밀도 | 파일 | 크기 | 비고 |
|--------|------|-----:|------|
| FP32 | `weights/yolo26n_drone_640_fp32.onnx` | 9.80 MB | 기준; opset17, static, simplified |
| FP16 | `weights/yolo26n_drone_640_fp16.onnx` | 4.97 MB | native `half=True`; float16 I/O |
| INT8 | `weights/yolo26n_drone_640_int8.onnx` | **3.01 MB** | static PTQ(QDQ), Conv-only, 200장 캘리브 |

**INT8 vs FP32** (동일 val 20장, conf 0.25): 탐지 **27→27**, 전부 IoU≥0.5 매칭, 평균 IoU 0.961,
평균 |Δscore| 0.075 → 저하 미미.

비교군 **yolo26s**도 동일 경로 export: FP32 38.2MB / FP16 19.2MB / INT8 10.2MB
(`weights/yolo26s_drone_640_{fp32,fp16,int8}.onnx`).

### Dev-CPU 지연 (방향성 추정치) — `weights/latency_report.md`

> ⚠️ **x86-64 데스크톱 CPU**(i9-13900K) ORT / CPUExecutionProvider 측정. 모바일 CPU의 *방향성*
> 추정치이며 실측 **아님**. 최종 수치는 타깃 디바이스 on-device 프로파일링 필요.

| 정밀도 | threads=1 (ms) | threads=4 (ms) | 크기 |
|--------|---------------:|---------------:|-----:|
| FP32 | 41.9 ± 1.5 | 12.9 ± 0.5 | 9.80 MB |
| FP16 | 42.9 ± 0.9 | 13.4 ± 0.3 | 4.97 MB |
| INT8 | **30.2 ± 0.9** | 14.1 ± 0.4 | **3.01 MB** |

INT8: 크기·단일 스레드 지연 유리. 4 스레드는 Conv-only QDQ dequant 오버헤드로 x86 격차 축소.
FP16: CPU 속도 이득 없음(ORT CPU에 native fp16 커널 없음) → 크기/이식성 옵션.

---

## 모델 명세 (I/O 계약)

ONNX를 추론 엔진에 통합할 때 필요한 입출력 계약이다 (imgsz 640 모델 기준; 960 변형은 입력·좌표가 960).

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
           eval.py  predict.py  export.py  bench_latency.py
configs/   dut_drone.yaml
weights/   yolo26{n,s}_drone_640.pt  yolo26n_drone_640_{fp32,fp16,int8}.onnx
           metrics.json  latency_report.md
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
| 지연 벤치 | `... python scripts/bench_latency.py --stem yolo26n_drone_640` | `python scripts/bench_latency.py ...` |
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
