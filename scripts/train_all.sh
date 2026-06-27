#!/usr/bin/env bash
# Full ML2-baseline training: yolo26n then yolo26s, 150 epochs each at imgsz 640.
# Sequential so they share the GPU. Each run auto-saves runs/<name>/{results.csv,weights/best.pt}.
# Usage:  bash scripts/train_all.sh            (native venv)
#         docker compose run --rm dronear bash scripts/train_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."

# Activate venv if present (no-op inside the Docker image, which has ultralytics globally).
[ -f .venv/bin/activate ] && source .venv/bin/activate

EPOCHS="${EPOCHS:-150}"
IMGSZ="${IMGSZ:-640}"

echo "=== [1/2] yolo26n  ${EPOCHS}ep imgsz${IMGSZ} ==="
python scripts/train.py --model yolo26n.pt --epochs "$EPOCHS" --imgsz "$IMGSZ" \
  --name "yolo26n_drone_${IMGSZ}"

echo "=== [2/2] yolo26s  ${EPOCHS}ep imgsz${IMGSZ} ==="
python scripts/train.py --model yolo26s.pt --epochs "$EPOCHS" --imgsz "$IMGSZ" \
  --name "yolo26s_drone_${IMGSZ}"

echo "=== DONE. Best weights ==="
ls -la runs/yolo26n_drone_${IMGSZ}/weights/best.pt runs/yolo26s_drone_${IMGSZ}/weights/best.pt
