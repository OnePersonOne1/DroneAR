# HANDOFF — D-FINE-L@960 test-set 평가 (GPU 4090×1)

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
