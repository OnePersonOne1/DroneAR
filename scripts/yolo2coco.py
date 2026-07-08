"""YOLO(txt) → COCO(json) 변환 — D-FINE 등 COCO 포맷 학습기용.

merged_drone의 각 split(train/val/test_dut/test_maciullo)에 대해
annotations/<split>.json 생성. 이미지는 이동/복사 없음(file_name = 이미지 파일명).

usage: python scripts/yolo2coco.py [--root /mnt/ssd_0/dataset/merged_drone]
"""
import argparse
import json
from pathlib import Path

from PIL import Image

SPLITS = ["train", "val", "test_dut", "test_maciullo"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def convert(root: Path, split: str) -> dict:
    img_dir = root / "images" / split
    lbl_dir = root / "labels" / split
    images, annotations = [], []
    ann_id = 1
    files = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    for img_id, p in enumerate(files, 1):
        with Image.open(p) as im:
            w, h = im.size
        images.append(dict(id=img_id, file_name=p.name, width=w, height=h))
        lbl = lbl_dir / f"{p.stem}.txt"
        if lbl.exists():
            for line in lbl.read_text().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cx, cy, bw, bh = (float(v) for v in parts[1:5])
                x, y = (cx - bw / 2) * w, (cy - bh / 2) * h
                annotations.append(dict(
                    id=ann_id, image_id=img_id, category_id=0,
                    bbox=[round(x, 2), round(y, 2), round(bw * w, 2), round(bh * h, 2)],
                    area=round(bw * w * bh * h, 2), iscrowd=0))
                ann_id += 1
    return dict(
        info=dict(description=f"merged_drone {split} (YOLO->COCO)"),
        categories=[dict(id=0, name="drone")],
        images=images, annotations=annotations)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/ssd_0/dataset/merged_drone")
    a = ap.parse_args()
    root = Path(a.root)
    out_dir = root / "annotations"
    out_dir.mkdir(exist_ok=True)
    for split in SPLITS:
        coco = convert(root, split)
        out = out_dir / f"{split}.json"
        out.write_text(json.dumps(coco))
        print(f"{split}: images={len(coco['images'])} anns={len(coco['annotations'])} -> {out}")


if __name__ == "__main__":
    main()
