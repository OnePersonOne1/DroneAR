# HANDOFF — 최종 통합 (GPU 4090×1): D-FINE-L 평가 → 지표 통일 → README 총정리 → 가중치 Drive 배포

> 구성: **A** D-FINE-L test 평가 · **B** 전 모델 지표 통일(COCO 마스터표) · **C** README 총정리 · **D** 가중치 Google Drive 배포.
> A를 먼저 끝내고 같은 세션에서 B→C→D로 이어간다.

## A. D-FINE-L@960 test-set 평가

목적: 학습 산출물 **`best_stg2.pth`**(DUT-val best, ep115)를 **held-out test(DUT·Maciullo)**에서
`scripts/dfine_eval.py`로 평가 → repo 상단 **COCO 통일 비교표**(yolo26n C/D · D-FINE-N)와 **동일 평가기(faster-coco-eval)**.
결과 파일 = `reports/dfine_l960_eval.json`.

> 배경: 학습 로그의 `test_coco_eval_bbox`(AP 0.7346)는 **DUT-val**(학습 검증 분할, DUT만·Maciullo는 train에만 — `reports/merge_stats.json` 확인)이라
> 상단 test 비교표와 split·평가기가 달라 직접 비교 불가. 이 평가로 **동일 조건 test 수치**를 만들어 D-FINE-N 행 옆에 D-FINE-L 행을 추가한다.

## 전제 (네트워크 볼륨에 이미 존재)
- 가중치: `/workspace/runs/merged_dfine_l_960/best_stg2.pth` (477MB)
- 데이터셋 tar: `/workspace/datasets/merged_drone.tar` (6.2GB, splits: train/val/test_dut/test_maciullo + labels)
- repo: `/workspace/DroneAR` (동일 볼륨 마운트 시 그대로 있음. 다른 볼륨이면 `git clone` 후 `git pull`)
- eval 스크립트: `DroneAR/scripts/dfine_eval.py` — ⚠️ `ROOT`가 `/mnt/ssd_0/dataset/merged_drone`로 **하드코딩**(28행), `device` 기본 `cuda`, `SPLITS=[test_dut, test_maciullo]`.

## 실행 절차 (RunPod PyTorch 템플릿 = torch/CUDA 기본 탑재 가정)

```bash
# 0) repo 최신화
cd /workspace/DroneAR && git pull

# 1) 데이터셋 추출 (전체; test만 필요하면 아래 주석 참고). /workspace = 네트워크볼륨(영구)
tar -xf /workspace/datasets/merged_drone.tar -C /workspace
#  → /workspace/merged_drone/{annotations,images,labels}/...
#  (test만: tar -xf ...merged_drone.tar -C /workspace \
#      merged_drone/annotations merged_drone/images/test_dut merged_drone/images/test_maciullo \
#      merged_drone/labels/test_dut merged_drone/labels/test_maciullo )

# 2) 스크립트의 하드코딩 ROOT(/mnt/ssd_0/dataset/merged_drone)를 실제 경로로 심링크
mkdir -p /mnt/ssd_0/dataset
ln -sfn /workspace/merged_drone /mnt/ssd_0/dataset/merged_drone

# 3) D-FINE 소스 + deps (torch 있으면 아래 pip만)
cd /root && [ -d D-FINE ] || git clone --depth 1 https://github.com/Peterande/D-FINE.git
cd /root/D-FINE
pip install -q faster-coco-eval pycocotools PyYAML scipy loguru transformers calflops opencv-python-headless

# 4) L 아키텍처 config 배치 (학습 때와 동일 구성 — include 해석용)
mkdir -p configs/dataset configs/dfine/custom
cp /workspace/merged_drone_detection_pod.yml configs/dataset/merged_drone_detection.yml   # 경로 이미 /workspace/merged_drone
cp /workspace/dfine_l960_3gpu_pod.yml       configs/dfine/custom/dfine_l960_merged.yml

# 5) 평가 실행 (DroneAR에서; --config 은 --dfine-root 기준 상대경로, imgsz=960 고정)
cd /workspace/DroneAR
python scripts/dfine_eval.py \
  --dfine-root /root/D-FINE \
  --config configs/dfine/custom/dfine_l960_merged.yml \
  --ckpt /workspace/runs/merged_dfine_l_960/best_stg2.pth \
  --imgsz 960 --pre square --conf 0.25 \
  --out reports/dfine_l960_eval
```

- 소요: 1×4090에서 test 4,827장 · **수 분** 예상. VRAM 24GB로 충분(960 추론).
- 산출: `reports/dfine_l960_eval.json` — `{meta, test_dut{AP50,AP50_95,far_recall,recall_by_bin,FP_per_image,...}, test_maciullo{...}}` (D-FINE-N `reports/dfine_n_eval.json`와 동일 스키마).

## 결과 반영 (README COCO 통일표에 D-FINE-L 행 추가)

`README.md`의 **"정확도 (held-out test) — AP는 전부 COCO eval로 통일"** 표(D-FINE-N 행 아래)에 추가:

```
| **D-FINE-L 120ep** | <DUT.AP50> / <DUT.AP50_95> | <DUT.far_recall> | <DUT.recall_by_bin["<8"]> | <Maci.AP50> / <Maci.AP50_95> | <Maci.far_recall> | <DUT.FP_per_image> · <Maci.FP_per_image> |
```
값은 `reports/dfine_l960_eval.json`에서 채움. 그런 다음 커밋·푸시:
```bash
git add reports/dfine_l960_eval.json README.md
git commit -m "feat: D-FINE-L@960 test 평가 — COCO 통일표에 행 추가"
git push origin main
```

## 주의 / 해석
- **직접 비교 대상**: 통일표의 **D-FINE-N·yolo26n C/D**는 faster-coco-eval → D-FINE-L 행과 동일 평가기라 **바로 비교 가능**.
- **yolo26l-P2(H·I행)**은 [Ablation 표](reports/ablation_matrix.md)에 있고 **ultralytics val 평가기**(COCO와 ~0.5pt 관례 차) → D-FINE-L(COCO)과 정밀 비교하려면 yolo26l도 faster-coco-eval 재측정 필요(C/D가 그렇게 통일된 전례). 러프 비교는 가능하나 이 각주를 붙일 것.
- **추론 해상도**: D-FINE 계열은 학습 스케일 특화 → **960 고정**. 1280/직사각은 붕괴(README D-FINE-N 분석 참조). 필요 시 `--rect 736 1280` 또는 `--imgsz 1280`로 검증만.
- ckpt 로딩: 스크립트가 `state["ema"]["module"]` 우선 사용(EMA). state_dict assert 실패 시 ckpt 키 구조 확인.

작성: RunPod 세션 검증 기반(config include·경로·출력 스키마 확인 완료). best_stg2 외 `best_stg1.pth`·`last.pth`·`checkpoint00XX.pth`도 `/workspace/runs/merged_dfine_l_960/`에 보관.

위 5절(A)의 D-FINE-L test 평가가 끝났다는 전제로, 아래 B·C·D를 **한 세션에서 이어서** 수행한다(모든 학습 완료 → 최종 통합).

---

# B. 전체 지표 통일 (모든 모델 → 동일 데이터셋·평가기·지표)

## 문제
현재 README에 정확도 표가 **3종**이고 평가기가 제각각이라 모델 간 직접 비교 불가:
| 표 | 위치 | 평가기 | 커버 모델 |
|---|---|---|---|
| 정확도 150ep | `README#정확도-150-epochs` | ultralytics val() | yolo26n/s @640/960 (val+test) |
| Ablation A~I | `reports/ablation_matrix.md` | ultralytics val() | yolo26n(A~G)·yolo26l(H·I) |
| COCO 통일표 | `README#정확도-held-out-test` | **faster-coco-eval** | yolo26n C/D · D-FINE-N |

→ **목표: 단일 마스터표.** 모든 모델을 **DUT-test·Maciullo-test**(held-out 원본)에서 **faster-coco-eval**로, **동일 지표**로 재측정.

## 핵심 갭
`faster-coco-eval` + 세밀 size-bin(`<8·8-16·16-24·24-32·32-64·64-128·128+`)을 산출하는 스크립트는 **현재 `scripts/dfine_eval.py`뿐**(D-FINE 전용). `eval_compare.py`는 ultralytics val() + 거친 bin(32/96px), `eval.py`는 ultralytics val→metrics.json. **yolo용 동일 스키마 평가기가 없다.**

## 할 일 1 — 통합 평가기 `scripts/unified_eval.py` 신설
`dfine_eval.py`의 **COCO/bin/FP 집계 블록(공통부)** 을 재사용하고, 추론 프론트엔드만 계열별로 분기:
- 공통(그대로 이식): `BINS`/`BIN_LABELS`, greedy IoU 매칭, `far_recall = <8 + 8-16`(=<16px), `<8px` recall, `FP_per_image`, `COCOeval_faster`로 AP50/AP50-95. **출력 스키마 = `dfine_eval.py`와 100% 동일** → 한 표로 합류.
- `--family dfine`: 기존 `dfine_eval.py` 경로(config+ckpt, `state["ema"]`).
- `--family yolo`: `ultralytics.YOLO(w).predict(source=img_dir, imgsz, conf, iou=0.7, stream=True, batch=16)` → 박스를 **원본 px**로 받아 동일 COCO-det/bin 집계에 투입(전처리·conf·IoU-match 상수 동일).
- 상수 고정(전 모델 공통): `--conf 0.25 --iou-match 0.5`, size@640 기준 bin, `--pre square`. imgsz는 **모델 학습 해상도**(아래 매트릭스).
- 출력: `reports/unified/<model_key>.json`.

> 구현 주의: yolo far/bin이 기존 `reports/dfine_n_eval.json`의 D-FINE 규칙과 1:1 일치하도록 `load_gt`(GT side@640 = `sqrt(w_norm·h_norm)×640`)를 `dfine_eval.py`와 **동일 정의**로 쓸 것. 두 스크립트의 bin 경계·매칭이 어긋나면 통일 실패.

## 할 일 2 — 전 모델 평가 실행 (모델 매트릭스)
| model_key | family | weights | imgsz | 비고(ablation) |
|---|---|---|---:|---|
| yolo26n_640_dut | yolo | `weights/yolo26/yolo26n_drone_640.pt` | 640 | A (old, DUT-only) |
| yolo26n_960_dut | yolo | `weights/yolo26/yolo26n_drone_960.pt` | 960 | B (DUT-only) |
| yolo26s_640 | yolo | `weights/yolo26/yolo26s_drone_640.pt` | 640 | — |
| yolo26s_960 | yolo | `weights/yolo26/yolo26s_drone_960.pt` | 960 | — |
| yolo26n_640_m100 | yolo | `weights/yolo26/yolo26n_drone_640_mergedataset_100epoch.pt` | 640 | **C** |
| yolo26n_640_m300 | yolo | `weights/yolo26/yolo26n_drone_640_mergedataset_300epoch.pt` | 640 | **D** |
| yolo26nP2_960_m100 | yolo | `weights/yolo26/yolo26n_drone_960p2_mergedataset_100epoch.pt` | 960 | **E/F** |
| yolo26lP2_960_m100 | yolo | `weights/yolo26/yolo26l_drone_960p2_mergedataset_100epoch.pt` | 960 | **H/I** |
| dfine_n_640_m220 | dfine | `weights/d_fine/dfine_n_drone_640_mergedataset_220epoch.pth` | 640 | D-FINE-N |
| dfine_l_960_m120 | dfine | `runs/merged_dfine_l_960/best_stg2.pth` | 960 | **D-FINE-L (신규)** |

```bash
# yolo (예)
python scripts/unified_eval.py --family yolo --key yolo26lP2_960_m100 \
  --weights weights/yolo26/yolo26l_drone_960p2_mergedataset_100epoch.pt --imgsz 960
# dfine (예)
python scripts/unified_eval.py --family dfine --key dfine_l_960_m120 \
  --dfine-root /root/D-FINE --config configs/dfine/custom/dfine_l960_merged.yml \
  --ckpt /workspace/runs/merged_dfine_l_960/best_stg2.pth --imgsz 960
# 10개 model_key 전부 반복 → reports/unified/*.json
```

## 할 일 3 — 마스터표 생성 `scripts/build_master_table.py`
`reports/unified/*.json`을 읽어 README용 md 생성. **컬럼(지표 보존):**
```
| 모델 | train | imgsz | DUT AP50 | DUT AP50-95 | DUT far(<16px) | DUT <8px | DUT FP/img | Maci AP50 | Maci AP50-95 | Maci far(<16px) | Maci <8px | Maci FP/img |
```
- 전체 size-bin recall(`<8`~`128+`)은 표엔 안 넣고 `reports/unified/*.json`에 보존(원하면 접이식 상세표).
- 정렬: 계열·크기순(yolo26n→s→l, P2, D-FINE-N→L).

# C. README.md 총정리 (구조 재편)

**교체·통합:**
- `## 성능 지표` 아래 **정확도 150ep 표**(L25)와 **COCO 통일표**(L378) → **단일 마스터표(B-3)로 대체**. 마스터표에 각주 1줄: "전 모델 동일 test·faster-coco-eval·conf0.25/IoU0.5".
- **Ablation 매트릭스**(`reports/ablation_matrix.md` + README L109): AP 컬럼을 **마스터표(COCO) 수치로 교체** → "ultralytics val ~0.5pt 차" 각주 **삭제**(더 이상 혼재 아님). 표의 **변수분리 해석(비교쌍 A→C 등)** 서술은 그대로 보존.
- **D-FINE-N COCO 통일표**(L378) → 마스터표에 흡수, 섹션엔 해석·정성예시만 남김.
- **D-FINE-L 섹션**(L441): DUT-val 수렴표는 "학습 곡선 요약"으로 유지, **비교 수치는 마스터표(DUT-test/Maci-test)로 이동**. val≠test 각주 유지.

**보존(그대로):** 속도표(GPU/CPU/Export, L40~99) · Maciullo 라벨 감사(L136) · 정성 예시(win examples) · 추론 입력 가이드(L462) · 구성·입력 특이점 비교(L405) · 추론 해상도 표(L417).

**중복 제거 체크:** 같은 수치가 2곳 이상 하드코딩된 곳 없애고 마스터표를 단일 출처(SoT)로. 로드맵의 완료/진행 목록 최신화.

# D. 가중치 Google Drive 배포

## 인벤토리
| 파일 | 모델 | 크기 | 현재 위치 | 조치 |
|---|---|---:|---|---|
| yolo26n_drone_640.pt 외 yolo `.pt` ×8 | yolo26n/s/l | 5~50MB | repo `weights/` | 유지 + Drive 미러 |
| dfine_n_drone_640_mergedataset_220epoch.pth | D-FINE-N | 58MB | repo `weights/` | 유지 + Drive 미러 |
| **best_stg2.pth** | **D-FINE-L(릴리스)** | **477MB** | `runs/merged_dfine_l_960/` | **Drive 전용**(>100MB) |
| best_stg1.pth · last.pth | D-FINE-L(보조) | 477MB×2 | 〃 | Drive 전용(선택) |
| checkpoint0107.pth 등 | D-FINE-L(스냅샷) | 477MB | 〃 | 로컬 보관(배포 제외) |

## Drive 폴더 구조(권장)
```
DroneAR-weights/               (공유: 링크 있는 사람 보기)
├─ yolo26/   *.pt (+ onnx/ncnn 원하면)
├─ dfine/    dfine_n_...220epoch.pth
└─ dfine_l_960/  best_stg2.pth (필수) · best_stg1.pth · last.pth
```

## 업로드 (GPU 파드에서 rclone)
```bash
rclone config            # gdrive remote 생성(대화형, 1회)
rclone copy /workspace/runs/merged_dfine_l_960/best_stg2.pth gdrive:DroneAR-weights/dfine_l_960/ -P
rclone copy weights/ gdrive:DroneAR-weights/mirror/ --include '*.pt' --include '*.pth' -P
```
각 파일/폴더 **공유 링크 생성**(뷰어 권한) → README 표에 기입.

## README 가중치 표(신설 — 링크 형식)
`## 리포지토리 구조` 부근에 추가:
```
| 모델 | imgsz | 릴리스 파일 | 크기 | 다운로드 |
|---|---:|---|---:|---|
| D-FINE-L | 960 | best_stg2.pth | 477MB | [⬇ Drive](<LINK>) |
| D-FINE-N | 640 | dfine_n_..._220epoch.pth | 58MB | repo `weights/` · [⬇ Drive](<LINK>) |
| yolo26l-P2 | 960 | yolo26l_...100epoch.pt | 50MB | repo `weights/` · [⬇ Drive](<LINK>) |
| … | | | | |
```
- 정책: **<100MB는 repo 유지**(clone 편의) + Drive 미러, **D-FINE-L(477MB)만 Drive 전용**. repo 슬림화를 원하면 전량 Drive 이전 후 `weights/`는 링크표만 남기는 것도 가능(별도 결정).
- 기존 `README#가중치-배포`(D-FINE-L 자리표시자 `<!-- DRIVE_LINK -->`)를 실제 링크로 교체.

## 마무리 커밋
```bash
git add scripts/unified_eval.py scripts/build_master_table.py reports/unified/ README.md reports/ablation_matrix.md
git commit -m "feat: 전 모델 지표 통일(COCO 마스터표) + README 총정리 + 가중치 Drive 링크"
git push origin main
```

> 검증 팁: 통일 후 **D-FINE-N 행이 기존 `reports/dfine_n_eval.json`(DUT 0.951/0.705, Maci 0.866/0.428)과 일치**하는지 확인 = 통합 평가기가 기존 D-FINE 경로와 동일함을 보장하는 회귀 체크.
