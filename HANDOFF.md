# HANDOFF — 새 환경 재구축

GitHub(코드·**모든 최종 가중치**·문서) + 이 문서면 어떤 머신에서도 재구축된다.
로컬 전용(`runs/` 중간 체크포인트·시각화·로그)은 resume/재생성용이라 **불필요**(학습 완료 상태).

## 0. 전제 경로

- 작업: `/mnt/ssd_0/workspace/DroneAR` (git clone)
- 데이터: `/mnt/ssd_0/dataset/` (아래 스크립트로 생성) — 경로 다르면 `configs/*.yaml`의 `path:` 수정.

## 1. 환경

```bash
git clone https://github.com/OnePersonOne1/DroneAR.git && cd DroneAR
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128  # 호스트 CUDA 맞춤
pip install -r requirements.txt
```
- torch는 **cu128**(호스트 CUDA 12.8). ultralytics 기본은 cu130 끌어와 "driver too old" → 위 인덱스 고정.
- Docker 대안: `docker compose pull` (이미지 `hanmyeongil/yolo26:v1`).

## 2. 데이터셋 재현

```bash
# (a) DUT-Anti-UAV — 원본 수동 다운로드: https://github.com/wangdongdut/DUT-Anti-UAV
python scripts/voc2yolo.py --src <DUT_VOC_root> --dst /mnt/ssd_0/dataset/dut_yolo
# (b) Maciullo — HF 미러 자동 취득
python scripts/fetch_maciullo.py --dst /mnt/ssd_0/dataset/DroneDetection
# (c) leakage-safe 병합 (seed 0 고정, split_mapping.csv는 repo에 커밋됨)
python scripts/merge_datasets.py   # → /mnt/ssd_0/dataset/merged_drone (심링크)
```
- merged_drone은 **심링크** — dut_yolo·DroneDetection 원본이 있어야 동작.
- 검증: `python scripts/analyze_merge.py` (train 56,646 / val 2,600 / test_dut 2,200 / test_maciullo 2,625).

## 3. YOLO26 학습·평가

```bash
# 예: merged 300ep (온디바이스 배포 모델 D)
python scripts/train.py --data configs/merged_drone.yaml --model yolo26n.pt \
  --imgsz 640 --epochs 300 --batch 64 --seed 0
# 평가(구 vs 신, 실패모드 포함)
python scripts/eval_compare.py --old /nonexistent.pt --new <best.pt> --imgsz 640
# far-recall 크기 bin 분석
python scripts/analyze_fn.py --models "name:<best.pt>" --imgsz 640
```
- 배포(ncnn-Vulkan/onnx): `scripts/export.py`, `scripts/parity_ncnn.py` — 상세 `README_ML2_Vulkan.md`.

## 4. D-FINE (DETR 계열) 학습·평가

```bash
# D-FINE 레포 별도 clone + 추가 deps
cd /mnt/ssd_0/workspace && git clone https://github.com/Peterande/D-FINE.git && cd D-FINE
pip install faster-coco-eval PyYAML tensorboard scipy calflops loguru transformers onnxscript onnxsim
# COCO ckpt (tuning 시작점)
wget https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_n_coco.pth

# YOLO 라벨 → COCO json (D-FINE 입력)
cd /mnt/ssd_0/workspace/DroneAR && python scripts/yolo2coco.py   # → merged_drone/annotations/*.json
# 설정은 repo에 커밋됨: configs/dfine/{dfine_hgnetv2_n_merged_drone,merged_drone_detection}.yml
#   → D-FINE/configs/ 로 복사 후 학습
cp configs/dfine/*.yml /mnt/ssd_0/workspace/D-FINE/configs/dfine/custom/  # detection.yml은 dataset/로
cd /mnt/ssd_0/workspace/D-FINE
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train.py \
  -c configs/dfine/custom/dfine_hgnetv2_n_merged_drone.yml -t dfine_n_coco.pth --seed 0 --use-amp
# 재개: -t 대신 -r <last.pth>

# 평가 (AP=COCO eval, far/FP=동일 greedy 프로토콜)
cd /mnt/ssd_0/workspace/DroneAR
python scripts/dfine_eval.py --config configs/dfine/custom/dfine_hgnetv2_n_merged_drone.yml \
  --ckpt weights/dfine_n_drone_640_mergedataset_220epoch.pth --imgsz 640
```
- N: merged·640·batch 32·220ep. 릴리스 = **ep191 EMA**(best_stat은 EMA 미추적 → 로그 재분석 선택).
- VRAM 팁: N batch 32(~10GB). L@960 batch 8(~17.6GB), batch 16 OOM.

## 5. 현재 상태 / 다음 (2026-07-09)

- 완료: yolo26 n/s/l-P2, D-FINE-N@640 — 결과·비교표·그래프 전부 README `## Ablation`·`## D-FINE` 섹션.
- **진행 예정: D-FINE-L@960** (RunPod 4×4090, yolo26l-P2 레시피 동등 비교). 패키지·절차 = `runpod_package/`(로컬, 미커밋) 또는 §4를 960 설정으로. 추론 **960 고정**(DETR은 학습 해상도 이탈 시 붕괴 — README 스윕표).
- 배포 방침: 온디바이스(ML2)=yolo26n(ncnn-Vulkan), 클라우드=D-FINE-L(예정).
- 보류: hard-negative(NEG_DIR), temporal(detect-then-track).
