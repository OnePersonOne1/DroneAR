#!/usr/bin/env python3
"""Dataset diagnostics for the YOLO-format DUT-Anti-UAV tree.

Outputs (to <dst>/_viz/):
  - area_hist.png      histogram of normalized box area (sqrt(w*h) too)
  - sample_*.jpg       a few images with their boxes drawn
And prints small-object statistics that drive the imgsz / P2-head decision.

"Small object" is reported both COCO-style (absolute area at a reference imgsz)
and as normalized box side, so the share of tiny drones is explicit.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dst", default="/mnt/ssd_0/dataset/dut_yolo")
    ap.add_argument("--imgsz", type=int, default=640,
                    help="reference imgsz for absolute-pixel small-object stats")
    ap.add_argument("--samples", type=int, default=5)
    return ap.parse_args()


def main():
    a = parse_args()
    dst = Path(a.dst)
    viz = dst / "_viz"
    viz.mkdir(parents=True, exist_ok=True)

    areas, sides = [], []          # normalized area (w*h) and side sqrt(w*h)
    for split in ("train", "val", "test"):
        for txt in (dst / "labels" / split).glob("*.txt"):
            for line in txt.read_text().splitlines():
                p = line.split()
                if len(p) != 5:
                    continue
                w, h = float(p[3]), float(p[4])
                areas.append(w * h)
                sides.append((w * h) ** 0.5)

    n = len(areas)
    areas.sort()
    sides.sort()
    ref = a.imgsz

    def pct(frac_list, thr):
        return 100.0 * sum(1 for v in frac_list if v < thr) / len(frac_list)

    def q(sorted_list, p):
        return sorted_list[min(len(sorted_list) - 1, int(p * len(sorted_list)))]

    print(f"boxes={n}")
    print(f"normalized side sqrt(w*h): min={sides[0]:.4f} "
          f"p25={q(sides,.25):.4f} median={q(sides,.5):.4f} "
          f"p75={q(sides,.75):.4f} max={sides[-1]:.4f}")
    # COCO: small if area < 32^2 px. At ref imgsz that is a normalized area thresh.
    small_area = (32.0 / ref) ** 2
    med_area = (96.0 / ref) ** 2
    print(f"\nAt imgsz={ref} (COCO size bins on the square-letterboxed image):")
    print(f"  SMALL  (<32px side,  norm side <{32/ref:.4f}): {pct(sides, 32/ref):5.1f}%")
    print(f"  MEDIUM (32-96px):                              "
          f"{pct(sides,96/ref)-pct(sides,32/ref):5.1f}%")
    print(f"  LARGE  (>96px side,  norm side >{96/ref:.4f}): {100-pct(sides,96/ref):5.1f}%")
    print(f"\n  norm side < 0.02 (~13px@640): {pct(sides,0.02):5.1f}%")
    print(f"  norm side < 0.05 (~32px@640): {pct(sides,0.05):5.1f}%")
    print(f"  norm side < 0.10 (~64px@640): {pct(sides,0.10):5.1f}%")

    # Histograms.
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].hist(areas, bins=60, color="#3b76d0")
    ax[0].set_title("Normalized box area (w*h)")
    ax[0].set_xlabel("area"); ax[0].set_ylabel("count")
    ax[1].hist(sides, bins=60, color="#d06b3b")
    ax[1].axvline(32 / ref, color="k", ls="--", lw=1, label=f"32px@{ref}")
    ax[1].axvline(96 / ref, color="gray", ls="--", lw=1, label=f"96px@{ref}")
    ax[1].set_title("Normalized box side sqrt(w*h)")
    ax[1].set_xlabel("side"); ax[1].legend()
    fig.tight_layout()
    fig.savefig(viz / "area_hist.png", dpi=110)
    print(f"\nsaved {viz/'area_hist.png'}")

    # Sample visualizations (train, first few with boxes).
    drawn = 0
    for txt in sorted((dst / "labels" / "train").glob("*.txt")):
        lines = [l for l in txt.read_text().splitlines() if l.strip()]
        if not lines:
            continue
        img_path = dst / "images" / "train" / (txt.stem + ".jpg")
        if not img_path.exists():
            continue
        im = Image.open(img_path).convert("RGB")
        W, H = im.size
        d = ImageDraw.Draw(im)
        for l in lines:
            _, xc, yc, w, h = map(float, l.split())
            x1, y1 = (xc - w / 2) * W, (yc - h / 2) * H
            x2, y2 = (xc + w / 2) * W, (yc + h / 2) * H
            d.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
        im.save(viz / f"sample_{drawn:02d}_{txt.stem}.jpg")
        drawn += 1
        if drawn >= a.samples:
            break
    print(f"saved {drawn} sample visualizations to {viz}")


if __name__ == "__main__":
    main()
