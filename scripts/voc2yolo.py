#!/usr/bin/env python3
"""Convert the DUT-Anti-UAV PASCAL VOC dataset to YOLO format.

Source layout (read-only, never modified):
    <src>/{train,val,test}/{img,xml}
        img/*.jpg              images
        xml/*.xml              VOC annotations (same stem as image)

Output layout (new tree; images are symlinked to save space):
    <dst>/images/{train,val,test}/*.jpg   (symlinks into <src>)
    <dst>/labels/{train,val,test}/*.txt   (YOLO labels)

Box conversion (VOC absolute pixels -> YOLO normalized), with W=width, H=height:
    x_center = ((x_min + x_max) / 2) / W      w = (x_max - x_min) / W
    y_center = ((y_min + y_max) / 2) / H      h = (y_max - y_min) / H
All values are clamped to [0, 1]. Single class -> index 0 ("drone").
Degenerate boxes (w<=0 or h<=0) are skipped and counted. Images with no valid
object get an empty .txt (negative sample).
"""
import argparse
import os
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from PIL import Image

# Source split dir -> YOLO split name. Handles both 'valid' and 'val'.
SPLIT_MAP = {"train": "train", "valid": "val", "val": "val", "test": "test"}


def parse_args():
    ap = argparse.ArgumentParser(description="DUT VOC -> YOLO converter")
    ap.add_argument("--src", default="/mnt/ssd_0/dataset/DUT",
                    help="source DUT root (read-only)")
    ap.add_argument("--dst", default="/mnt/ssd_0/dataset/dut_yolo",
                    help="output YOLO root")
    ap.add_argument("--copy", action="store_true",
                    help="copy images instead of symlinking")
    return ap.parse_args()


def clamp01(v):
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


def convert_xml(xml_path, img_path):
    """Return (yolo_lines, n_boxes, n_skipped, class_names_seen)."""
    lines, n_boxes, n_skipped = [], 0, 0
    names = Counter()
    root = ET.parse(xml_path).getroot()

    size = root.find("size")
    W = H = None
    if size is not None:
        wv, hv = size.findtext("width"), size.findtext("height")
        if wv and hv and int(float(wv)) > 0 and int(float(hv)) > 0:
            W, H = int(float(wv)), int(float(hv))
    if W is None or H is None:               # fall back to actual image
        with Image.open(img_path) as im:
            W, H = im.size

    for obj in root.findall("object"):
        names[(obj.findtext("name") or "").strip() or "<empty>"] += 1
        bb = obj.find("bndbox")
        if bb is None:
            n_skipped += 1
            continue
        xmin = float(bb.findtext("xmin"))
        ymin = float(bb.findtext("ymin"))
        xmax = float(bb.findtext("xmax"))
        ymax = float(bb.findtext("ymax"))
        bw, bh = xmax - xmin, ymax - ymin
        if bw <= 0 or bh <= 0:
            n_skipped += 1
            continue
        xc = clamp01(((xmin + xmax) / 2.0) / W)
        yc = clamp01(((ymin + ymax) / 2.0) / H)
        nw = clamp01(bw / W)
        nh = clamp01(bh / H)
        if nw <= 0 or nh <= 0:               # clamp could zero a sliver
            n_skipped += 1
            continue
        lines.append(f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
        n_boxes += 1
    return lines, n_boxes, n_skipped, names


def link_or_copy(src_img, dst_img, do_copy):
    if dst_img.exists() or dst_img.is_symlink():
        dst_img.unlink()
    if do_copy:
        import shutil
        shutil.copy2(src_img, dst_img)
    else:
        os.symlink(os.path.abspath(src_img), dst_img)


def main():
    args = parse_args()
    src, dst = Path(args.src), Path(args.dst)
    if not src.is_dir():
        sys.exit(f"ERROR: source not found: {src}")

    grand = Counter()
    all_names = Counter()
    print(f"src={src}  dst={dst}  mode={'copy' if args.copy else 'symlink'}\n")

    for src_split, yolo_split in SPLIT_MAP.items():
        sdir = src / src_split
        if not (sdir / "img").is_dir():
            continue                          # e.g. both 'valid' and 'val' in map
        img_dir = sdir / "img"
        xml_dir = sdir / "xml"
        out_img = dst / "images" / yolo_split
        out_lbl = dst / "labels" / yolo_split
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)

        n_img = n_lbl = n_box = n_skip = n_neg = n_missing = 0
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue
            n_img += 1
            xml = xml_dir / (img.stem + ".xml")
            if xml.exists():
                lines, nb, nsk, names = convert_xml(xml, img)
                all_names.update(names)
                n_box += nb
                n_skip += nsk
            else:
                lines, n_missing = [], n_missing + 1
            if not lines:
                n_neg += 1
            (out_lbl / (img.stem + ".txt")).write_text("\n".join(lines))
            n_lbl += 1
            link_or_copy(img, out_img / img.name, args.copy)

        print(f"[{src_split:>5} -> {yolo_split:<5}] images={n_img}  labels={n_lbl}  "
              f"boxes={n_box}  skipped={n_skip}  negatives={n_neg}  missing_xml={n_missing}")
        grand.update({"images": n_img, "labels": n_lbl, "boxes": n_box,
                      "skipped": n_skip, "negatives": n_neg, "missing_xml": n_missing})

    print("\n=== TOTAL ===")
    for k in ("images", "labels", "boxes", "skipped", "negatives", "missing_xml"):
        print(f"  {k:12} {grand[k]}")
    print(f"  class names seen: {dict(all_names)}  -> all mapped to class 0 (drone)")
    print(f"\nDone. YOLO dataset at: {dst}")


if __name__ == "__main__":
    main()
