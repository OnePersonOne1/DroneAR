#!/usr/bin/env python3
"""Phase 1.6 — comparison analysis: DUT vs Maciullo vs merged.

(1) bbox-size distribution from YOLO labels:
      area_ratio = w_norm * h_norm                     (scale-invariant)
      side@640   = sqrt(w_norm*h_norm) * 640           (object size on the 640 canvas)
    Report percentiles + COCO-style small/med/large fractions (side<32 / 32..96 / >96).

(2) background composition proxy (does Maciullo actually add non-sky backgrounds?):
    sample images per source, gray -> resize 256, Sobel gradient magnitude mean =
    "edge_density". Sky/uniform backgrounds -> low; ground/urban/terrain clutter -> high.
    Report edge-density percentiles + fraction "ground-like" (>= cut).

Writes reports/dataset_comparison.md (+ .json).
"""
import argparse
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np


def load_boxes(label_dir: Path, prefix=None):
    """Return list of (area_ratio, side640) for all boxes under label_dir."""
    out = []
    for txt in label_dir.glob("*.txt"):
        if prefix and not txt.name.startswith(prefix):
            continue
        for ln in txt.read_text().splitlines():
            p = ln.split()
            if len(p) != 5:
                continue
            w, h = float(p[3]), float(p[4])
            ar = w * h
            out.append((ar, math.sqrt(max(ar, 0.0)) * 640))
    return out


def box_stats(boxes):
    if not boxes:
        return {}
    sides = np.array([b[1] for b in boxes])
    ars = np.array([b[0] for b in boxes])
    pct = lambda a, q: float(np.percentile(a, q))
    n = len(sides)
    return {
        "n_boxes": n,
        "side640_p": {q: round(pct(sides, q), 2) for q in (5, 25, 50, 75, 95)},
        "side640_mean": round(float(sides.mean()), 2),
        "area_ratio_median": round(float(np.median(ars)), 6),
        "frac_small_<32px": round(float((sides < 32).mean()), 4),
        "frac_med_32-96px": round(float(((sides >= 32) & (sides <= 96)).mean()), 4),
        "frac_large_>96px": round(float((sides > 96).mean()), 4),
    }


def edge_density(img_path: Path):
    im = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if im is None:
        return None
    im = cv2.resize(im, (256, 256), interpolation=cv2.INTER_AREA)
    gx = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    return float(mag.mean())


def bg_stats(img_dir: Path, sample, rng, prefix=None, glob="*.jpg"):
    files = [f for f in img_dir.glob(glob) if (not prefix or f.name.startswith(prefix))]
    rng.shuffle(files)
    files = files[:sample]
    vals = [v for f in files if (v := edge_density(f)) is not None]
    return np.array(vals), len(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", default="/mnt/ssd_0/dataset/merged_drone")
    ap.add_argument("--dut-yolo", default="/mnt/ssd_0/dataset/dut_yolo")
    ap.add_argument("--repo", default="/mnt/ssd_0/workspace/DroneAR")
    ap.add_argument("--sample", type=int, default=800)
    ap.add_argument("--ground-cut", type=float, default=12.0,
                    help="edge-density >= cut counts as ground/clutter-like")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    merged, dut = Path(a.merged), Path(a.dut_yolo)
    rng = random.Random(a.seed)

    # (1) bbox sizes
    dut_boxes = load_boxes(dut / "labels" / "train")
    maci_boxes = load_boxes(merged / "labels" / "train", prefix="train_")
    merged_boxes = load_boxes(merged / "labels" / "train")
    box = {"DUT_train": box_stats(dut_boxes),
           "Maciullo_train": box_stats(maci_boxes),
           "merged_train": box_stats(merged_boxes)}

    # (2) background proxy
    bg = {}
    for name, (img_dir, prefix) in {
        "DUT_train": (dut / "images" / "train", None),
        "Maciullo_train": (merged / "images" / "train", "train_"),
        "merged_train": (merged / "images" / "train", None),
    }.items():
        vals, k = bg_stats(img_dir, a.sample, random.Random(a.seed), prefix)
        if len(vals):
            bg[name] = {
                "sampled": k,
                "edge_density_p": {q: round(float(np.percentile(vals, q)), 2)
                                   for q in (25, 50, 75)},
                "edge_density_mean": round(float(vals.mean()), 2),
                "frac_ground_like": round(float((vals >= a.ground_cut).mean()), 4),
            }

    out = {"bbox_size": box, "background_proxy": bg,
           "ground_cut": a.ground_cut, "sample": a.sample}
    rep = Path(a.repo) / "reports"
    rep.mkdir(exist_ok=True)
    (rep / "dataset_comparison.json").write_text(json.dumps(out, indent=2))

    L = ["# Dataset comparison — DUT vs Maciullo vs merged (train split)", "",
         "> 생성물 — `scripts/analyze_merge.py` 재실행 시 덮어써짐.", "",
         "## bbox size distribution (from YOLO labels)", "",
         "| set | boxes | side@640 p50 | p5..p95 | small<32px | med | large>96px |",
         "|---|---:|---:|---|---:|---:|---:|"]
    for k, s in box.items():
        if not s:
            continue
        p = s["side640_p"]
        L.append(f"| {k} | {s['n_boxes']} | {p[50]} | {p[5]}..{p[95]} | "
                 f"{s['frac_small_<32px']} | {s['frac_med_32-96px']} | {s['frac_large_>96px']} |")
    L += ["", "## background composition proxy (Sobel edge density; sky=low, ground/clutter=high)",
          "", f"sample={a.sample}/set, ground-like cut = {a.ground_cut}", "",
          "| set | sampled | edge dens p50 | mean | frac ground-like |",
          "|---|---:|---:|---:|---:|"]
    for k, s in bg.items():
        L.append(f"| {k} | {s['sampled']} | {s['edge_density_p'][50]} | "
                 f"{s['edge_density_mean']} | {s['frac_ground_like']} |")
    L += ["", "## reading",
          "- If Maciullo shows **higher edge density / higher frac ground-like** than DUT,",
          "  it is adding non-sky (ground/terrain/urban) backgrounds — the intended lever",
          "  for reducing ground-background misses and ground-clutter false positives.",
          "- bbox-size overlap tells whether the merge shifts the object-scale regime",
          "  (relevant to small-object recall in Phase 4).",
          "- Proxy caveat: edge density is computed over the whole frame (drone + bg);",
          "  it is a coarse sky-vs-ground signal, not a segmentation."]
    (rep / "dataset_comparison.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nsaved {rep/'dataset_comparison.md'}")


if __name__ == "__main__":
    main()
