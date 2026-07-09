#!/usr/bin/env python3
"""D-FINE 체크포인트를 test_dut/test_maciullo에서 평가 — yolo26 지표와 동일 정의.

- AP50/AP50-95: COCO eval(faster-coco-eval) vs annotations/<split>.json — **프로토콜 주의**:
  ultralytics val()과 산출기 다름(101-pt 보간 등), 표 비교 시 주석 필요.
- far-recall/size-bin/FP: scripts/analyze_fn.py·eval_compare.py와 동일 규칙
  (side640 = sqrt(w_norm·h_norm)×640, conf 0.25, greedy IoU>=0.5).

usage:
  python scripts/dfine_eval.py --dfine-root /mnt/ssd_0/workspace/D-FINE \
    --config configs/dfine/custom/dfine_hgnetv2_n_merged_drone.yml \
    --ckpt runs/merged_dfine_n_640/best_stg1.pth --imgsz 640 \
    --out reports/dfine_n_eval
"""
import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image

BINS = [(0, 8), (8, 16), (16, 24), (24, 32), (32, 64), (64, 128), (10 ** 9, 10 ** 9)]
BINS[-1] = (128, 10 ** 9)
BIN_LABELS = ["<8", "8-16", "16-24", "24-32", "32-64", "64-128", "128+"]
ROOT = Path("/mnt/ssd_0/dataset/merged_drone")
SPLITS = ["test_dut", "test_maciullo"]


def iou_xyxy(a, b):
    xa, ya = max(a[0], b[0]), max(a[1], b[1])
    xb, yb = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    ua = max(0.0, a[2]-a[0])*max(0.0, a[3]-a[1]) + max(0.0, b[2]-b[0])*max(0.0, b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def bin_of(side):
    for i, (lo, hi) in enumerate(BINS):
        if lo <= side < hi:
            return i
    return len(BINS) - 1


def load_model(dfine_root, config, ckpt, imgsz, device):
    sys.path.insert(0, dfine_root)
    from src.core import YAMLConfig
    cfg = YAMLConfig(str(Path(dfine_root) / config) if not Path(config).is_absolute() else config)
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    cfg.yaml_cfg["eval_spatial_size"] = [imgsz, imgsz]
    model = cfg.model
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    weights = state["ema"]["module"] if "ema" in state else state["model"]
    # eval_spatial_size 의존 프리컴퓨트 버퍼는 새 해상도용(모델 생성값) 유지
    for k in ["decoder.anchors", "decoder.valid_mask"]:
        weights.pop(k, None)
    missing, unexpected = model.load_state_dict(weights, strict=False)
    assert not unexpected and all(k.startswith("decoder.") for k in missing), (missing, unexpected)
    model = model.deploy().to(device)
    post = cfg.postprocessor.deploy().to(device)
    model.eval()
    return model, post


@torch.no_grad()
def eval_split(model, post, split, imgsz, device, conf, iou_match, batch=16):
    img_dir, lbl_dir = ROOT / "images" / split, ROOT / "labels" / split
    ann = json.loads((ROOT / "annotations" / f"{split}.json").read_text())
    name2id = {im["file_name"]: im["id"] for im in ann["images"]}
    tf = T.Compose([T.Resize((imgsz, imgsz)), T.ToTensor()])
    files = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})

    coco_dets = []
    matched = [0] * len(BINS)
    total = [0] * len(BINS)
    tp = fp = 0
    for i in range(0, len(files), batch):
        chunk = files[i:i + batch]
        ims, sizes, metas = [], [], []
        for p in chunk:
            im = Image.open(p).convert("RGB")
            sizes.append([im.width, im.height])
            ims.append(tf(im))
            metas.append(p)
        x = torch.stack(ims).to(device)
        orig = torch.tensor(sizes, dtype=torch.int64, device=device)
        labels, boxes, scores = post(model(x), orig)
        for j, p in enumerate(metas):
            W, H = sizes[j]
            keep = scores[j] > conf
            dets = boxes[j][keep].cpu().tolist()
            scs = scores[j][keep].cpu().tolist()
            img_id = name2id[p.name]
            for d, s in zip(dets, scs):
                coco_dets.append(dict(image_id=img_id, category_id=0, score=s,
                                      bbox=[d[0], d[1], d[2]-d[0], d[3]-d[1]]))
            # far/size-bin: analyze_fn과 동일(greedy, conf 0.25)
            gts = []
            lp = lbl_dir / f"{p.stem}.txt"
            if lp.exists():
                for ln in lp.read_text().splitlines():
                    q = ln.split()
                    if len(q) != 5:
                        continue
                    cx, cy, w, h = (float(v) for v in q[1:])
                    gts.append([(cx-w/2)*W, (cy-h/2)*H, (cx+w/2)*W, (cy+h/2)*H,
                                math.sqrt(max(w*h, 0.0)) * 640])
            used = [False] * len(gts)
            order = sorted(range(len(dets)), key=lambda k: -scs[k])
            for k in order:
                best, bj = iou_match, -1
                for gj, g in enumerate(gts):
                    if used[gj]:
                        continue
                    v = iou_xyxy(dets[k], g[:4])
                    if v >= best:
                        best, bj = v, gj
                if bj >= 0:
                    used[bj] = True
                    tp += 1
                else:
                    fp += 1
            for gj, g in enumerate(gts):
                b = bin_of(g[4])
                total[b] += 1
                matched[b] += int(used[gj])

    # COCO AP (전체 score, conf 필터 없이가 표준이지만 postprocess 상위 300 유지)
    from faster_coco_eval import COCO, COCOeval_faster
    gt = COCO(str(ROOT / "annotations" / f"{split}.json"))
    dt = gt.loadRes(coco_dets) if coco_dets else None
    ev = COCOeval_faster(gt, dt, "bbox")
    ev.evaluate(); ev.accumulate(); ev.summarize()
    ap5095, ap50 = float(ev.stats[0]), float(ev.stats[1])

    n_img = len(files)
    far_m = matched[0] + matched[1]
    far_t = total[0] + total[1]
    return dict(
        AP50=round(ap50, 4), AP50_95=round(ap5095, 4),
        far_recall=round(far_m / far_t, 4) if far_t else None,
        recall_by_bin={l: (round(m / t, 4) if t else None)
                       for l, m, t in zip(BIN_LABELS, matched, total)},
        gt_by_bin=dict(zip(BIN_LABELS, total)),
        TP=tp, FP=fp, FP_per_image=round(fp / n_img, 4), images=n_img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dfine-root", default="/mnt/ssd_0/workspace/D-FINE")
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou-match", type=float, default=0.5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="reports/dfine_eval")
    a = ap.parse_args()

    model, post = load_model(a.dfine_root, a.config, a.ckpt, a.imgsz, a.device)
    result = {}
    for split in SPLITS:
        result[split] = eval_split(model, post, split, a.imgsz, a.device, a.conf, a.iou_match)
        print(split, json.dumps(result[split], indent=1))
    meta = dict(ckpt=a.ckpt, imgsz=a.imgsz, conf=a.conf, iou_match=a.iou_match,
                note="AP=COCO eval(faster-coco-eval) — ultralytics val()과 산출기 다름")
    Path(a.out + ".json").write_text(json.dumps(dict(meta=meta, **result), indent=2))
    print("saved", a.out + ".json")


if __name__ == "__main__":
    main()
