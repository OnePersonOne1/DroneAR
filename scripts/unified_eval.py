#!/usr/bin/env python3
"""통합 평가기 — 전 모델(yolo·D-FINE)을 동일 데이터셋·평가기·지표로 재측정.

held-out test(DUT-test·Maciullo-test) · faster-coco-eval · 세밀 size-bin.
출력 스키마 = `scripts/dfine_eval.py`와 100% 동일 → 한 마스터표로 합류.
- D-FINE 계열: `dfine_eval.eval_split`을 그대로 호출(회귀 일관성 보장).
- yolo 계열: ultralytics predict → 원본 px 박스를 동일 COCO/bin/FP 집계에 투입.

상수(전 모델 공통): conf 0.25 · IoU-match 0.5 · side@640 bin · D-FINE는 square 전처리.
imgsz는 모델 학습 해상도. 출력: reports/unified/<key>.json.

usage:
  # yolo
  python scripts/unified_eval.py --family yolo --key yolo26lP2_960_m100 \
    --weights weights/yolo26/yolo26l_drone_960p2_mergedataset_100epoch.pt --imgsz 960
  # dfine
  python scripts/unified_eval.py --family dfine --key dfine_l_960_m120 \
    --dfine-root /root/D-FINE --config configs/dfine/custom/dfine_l960_merged.yml \
    --ckpt /workspace/runs/merged_dfine_l_960/best_stg2.pth --imgsz 960
"""
import argparse
import json
import math
import sys
from pathlib import Path

# dfine_eval의 공통 프리미티브 재사용(동일 정의 보장) — 같은 폴더
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dfine_eval import (  # noqa: E402
    BINS, BIN_LABELS, ROOT, SPLITS, iou_xyxy, bin_of,
    load_model as dfine_load_model, eval_split as dfine_eval_split,
)


def eval_split_yolo(model, split, imgsz, device, conf, iou_match, batch=16):
    """ultralytics YOLO — dfine_eval.eval_split과 동일 스키마 dict 반환.

    박스는 ultralytics 원본 px(res.boxes.xyxy). COCO-det/greedy 매칭/side640 bin은
    dfine_eval와 동일 규칙(conf 0.25 사전필터 = predict conf, greedy IoU>=0.5).
    """
    from faster_coco_eval import COCO, COCOeval_faster

    img_dir, lbl_dir = ROOT / "images" / split, ROOT / "labels" / split
    ann = json.loads((ROOT / "annotations" / f"{split}.json").read_text())
    name2id = {im["file_name"]: im["id"] for im in ann["images"]}
    files = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    n_img = len(files)

    coco_dets = []
    matched = [0] * len(BINS)
    total = [0] * len(BINS)
    tp = fp = 0
    for res in model.predict(source=str(img_dir), imgsz=imgsz, device=device,
                             conf=conf, iou=0.7, verbose=False, stream=True, batch=batch):
        p = Path(res.path)
        H, W = res.orig_shape
        img_id = name2id[p.name]
        if res.boxes is not None and len(res.boxes):
            dets = res.boxes.xyxy.cpu().tolist()
            scs = res.boxes.conf.cpu().tolist()
        else:
            dets, scs = [], []
        for d, s in zip(dets, scs):
            coco_dets.append(dict(image_id=img_id, category_id=0, score=s,
                                  bbox=[d[0], d[1], d[2]-d[0], d[3]-d[1]]))
        # far/size-bin: dfine_eval과 동일(greedy, conf 0.25, side640)
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

    gt = COCO(str(ROOT / "annotations" / f"{split}.json"))
    dt = gt.loadRes(coco_dets) if coco_dets else None
    ev = COCOeval_faster(gt, dt, "bbox")
    ev.evaluate(); ev.accumulate(); ev.summarize()
    ap5095, ap50 = float(ev.stats[0]), float(ev.stats[1])

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
    ap.add_argument("--family", required=True, choices=["yolo", "dfine"])
    ap.add_argument("--key", required=True, help="model_key → reports/unified/<key>.json")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou-match", type=float, default=0.5)
    ap.add_argument("--device", default="cuda")
    # yolo
    ap.add_argument("--weights", help="yolo .pt")
    # dfine
    ap.add_argument("--dfine-root", default="/root/D-FINE")
    ap.add_argument("--config")
    ap.add_argument("--ckpt")
    ap.add_argument("--pre", choices=["square", "letterbox"], default="square")
    a = ap.parse_args()

    result = {}
    if a.family == "yolo":
        assert a.weights, "--weights 필요"
        from ultralytics import YOLO
        dev = a.device if a.device != "cuda" else "0"
        model = YOLO(a.weights)
        for split in SPLITS:
            result[split] = eval_split_yolo(model, split, a.imgsz, dev, a.conf, a.iou_match)
            print(split, json.dumps(result[split], indent=1))
        meta = dict(family="yolo", key=a.key, weights=a.weights, imgsz=a.imgsz,
                    conf=a.conf, iou_match=a.iou_match,
                    note="AP=COCO eval(faster-coco-eval), conf 0.25 사전필터 · yolo letterbox 전처리(학습 동일)")
    else:
        assert a.config and a.ckpt, "--config·--ckpt 필요"
        size_hw = (a.imgsz, a.imgsz)
        model, post = dfine_load_model(a.dfine_root, a.config, a.ckpt, size_hw, a.device)
        for split in SPLITS:
            result[split] = dfine_eval_split(model, post, split, size_hw, a.device,
                                             a.conf, a.iou_match, pre=a.pre)
            print(split, json.dumps(result[split], indent=1))
        meta = dict(family="dfine", key=a.key, ckpt=a.ckpt, config=a.config, imgsz=a.imgsz,
                    pre=a.pre, conf=a.conf, iou_match=a.iou_match,
                    note="AP=COCO eval(faster-coco-eval) — dfine_eval.eval_split 동일 경로")

    outdir = Path("reports/unified")
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"{a.key}.json"
    out.write_text(json.dumps(dict(meta=meta, **result), indent=2))
    print("saved", out)


if __name__ == "__main__":
    main()
