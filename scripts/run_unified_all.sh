#!/usr/bin/env bash
# 전 모델 통합 평가 드라이버 — reports/unified/*.json 생성.
set -euo pipefail
cd /workspace/DroneAR

run_yolo() { # key weights imgsz
  echo "===== [yolo] $1 (imgsz=$3) ====="
  python scripts/unified_eval.py --family yolo --key "$1" --weights "$2" --imgsz "$3"
}

run_yolo yolo26n_640_dut     weights/yolo26n_drone_640.pt                              640
run_yolo yolo26n_960_dut     weights/yolo26n_drone_960.pt                              960
run_yolo yolo26s_640         weights/yolo26s_drone_640.pt                              640
run_yolo yolo26s_960         weights/yolo26s_drone_960.pt                              960
run_yolo yolo26n_640_m100    weights/yolo26n_drone_640_mergedataset_100epoch.pt        640
run_yolo yolo26nP2_960_m100  weights/yolo26n_drone_960p2_mergedataset_100epoch.pt      960
run_yolo yolo26lP2_960_m100  weights/yolo26l_drone_960p2_mergedataset_100epoch.pt      960

echo "===== [dfine] dfine_l_960_m120 (imgsz=960) ====="
python scripts/unified_eval.py --family dfine --key dfine_l_960_m120 \
  --dfine-root /root/D-FINE --config configs/dfine/custom/dfine_l960_merged.yml \
  --ckpt /workspace/runs/merged_dfine_l_960/best_stg2.pth --imgsz 960

echo "ALL_UNIFIED_DONE"
