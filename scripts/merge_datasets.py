#!/usr/bin/env python3
"""Phase 1 — unify to YOLO + leakage-safe merge (DUT + Maciullo).

Builds MERGED_DIR with a single-class (0=drone) YOLO layout:

    images/train          DUT-train symlinks  +  Maciullo-train symlinks
    images/val            DUT-val symlinks ONLY           (leakage-safe val)
    images/test_dut       DUT-test symlinks               (held-out eval A)
    images/test_maciullo  Maciullo-test symlinks          (held-out eval B)
    labels/<same>/*.txt   YOLO labels (empty .txt = negative/background)

Leakage rule (see DroneDetection/provenance.json): the HF Maciullo mirror has no
video_id, so a group split by sequence is impossible. Fallback that never crosses
the official train/test boundary: all Maciullo-train -> merged train; merged val is
DUT's official val ONLY; both original test sets are preserved untouched as separate
eval sets. No Maciullo frame is shared between train and val.

Inputs:
  DUT   : reuse existing YOLO tree from scripts/voc2yolo.py  (dut_yolo/{images,labels})
  Maci. : DroneDetection/{images,annotations} from scripts/fetch_maciullo.py

MAX_FRAMES_PER_VIDEO applies only if >0; with no video_id it degrades to a uniform
stride over Maciullo-train image order (logged as an approximation, off by default).
"""
import argparse
import json
import os
from pathlib import Path


def link(src: Path, dst: Path):
    src = Path(os.path.abspath(os.path.realpath(src)))
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    os.symlink(src, dst)


def coco_to_yolo(boxes, cats, W, H):
    """COCO xywh(px) -> YOLO 'cls cx cy w h' normalized, clamped, class 0."""
    out = []
    for (x, y, w, h), _c in zip(boxes, cats):
        if w <= 0 or h <= 0:
            continue
        cx, cy = (x + w / 2) / W, (y + h / 2) / H
        nw, nh = w / W, h / H
        cx = min(max(cx, 0.0), 1.0); cy = min(max(cy, 0.0), 1.0)
        nw = min(max(nw, 0.0), 1.0); nh = min(max(nh, 0.0), 1.0)
        if nw <= 0 or nh <= 0:
            continue
        out.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    return out


def add_maciullo(dst, maci, split, out_img, out_lbl, stride, mapping):
    ann = maci / "annotations" / f"{split}.jsonl"
    recs = [json.loads(l) for l in ann.read_text().splitlines() if l.strip()]
    if stride > 1:
        recs = recs[::stride]
    n_box = n_neg = 0
    for r in recs:
        img = maci / "images" / split / r["file"]
        stem = Path(r["file"]).stem
        link(img, out_img / r["file"])
        lines = coco_to_yolo(r["boxes"], r["category"], r["width"], r["height"])
        (out_lbl / f"{stem}.txt").write_text("\n".join(lines))
        n_box += len(lines)
        if not lines:
            n_neg += 1
        mapping.append((f"maciullo/{split}", r["file"], out_img.name, len(lines)))
    return len(recs), n_box, n_neg


def add_dut(dut_yolo, dut_split, out_img, out_lbl, mapping, dest_name):
    src_img = dut_yolo / "images" / dut_split
    src_lbl = dut_yolo / "labels" / dut_split
    n = n_box = n_neg = 0
    for jpg in sorted(src_img.glob("*.jpg")):
        stem = jpg.stem
        link(jpg, out_img / jpg.name)
        lbl = src_lbl / f"{stem}.txt"
        txt = lbl.read_text() if lbl.exists() else ""
        (out_lbl / f"{stem}.txt").write_text(txt)
        k = len([ln for ln in txt.splitlines() if ln.strip()])
        n += 1; n_box += k
        if k == 0:
            n_neg += 1
        mapping.append((f"dut/{dut_split}", jpg.name, dest_name, k))
    return n, n_box, n_neg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", default="/mnt/ssd_0/dataset/merged_drone")
    ap.add_argument("--dut-yolo", default="/mnt/ssd_0/dataset/dut_yolo")
    ap.add_argument("--maciullo", default="/mnt/ssd_0/dataset/DroneDetection")
    ap.add_argument("--max-frames-per-video", type=int, default=0)
    ap.add_argument("--repo", default="/mnt/ssd_0/workspace/DroneAR")
    a = ap.parse_args()
    merged, dut_yolo, maci = Path(a.merged), Path(a.dut_yolo), Path(a.maciullo)

    # dest split dirs
    dirs = {}
    for d in ("train", "val", "test_dut", "test_maciullo"):
        for kind in ("images", "labels"):
            p = merged / kind / d
            p.mkdir(parents=True, exist_ok=True)
            dirs[(kind, d)] = p

    stride = 1
    if a.max_frames_per_video > 0:
        # No video_id available; approximate redundancy control by uniform stride.
        stride = max(1, a.max_frames_per_video)  # interpreted as "keep 1 of every N"
        print(f"[warn] MAX_FRAMES_PER_VIDEO>0 but no video_id; applying uniform "
              f"stride keep-1-of-{stride} over Maciullo-train order (approximation).")

    mapping = []
    stats = {}

    # --- merged train = DUT train + Maciullo train ---
    d_n, d_b, d_neg = add_dut(dut_yolo, "train", dirs[("images", "train")],
                              dirs[("labels", "train")], mapping, "train")
    m_n, m_b, m_neg = add_maciullo(merged, maci, "train", dirs[("images", "train")],
                                   dirs[("labels", "train")],
                                   stride if a.max_frames_per_video > 0 else 1, mapping)
    stats["train"] = {"dut": [d_n, d_b, d_neg], "maciullo": [m_n, m_b, m_neg],
                      "total_images": d_n + m_n, "total_boxes": d_b + m_b,
                      "negatives": d_neg + m_neg}

    # --- merged val = DUT val ONLY (leakage-safe) ---
    v_n, v_b, v_neg = add_dut(dut_yolo, "val", dirs[("images", "val")],
                              dirs[("labels", "val")], mapping, "val")
    stats["val"] = {"dut": [v_n, v_b, v_neg], "total_images": v_n,
                    "total_boxes": v_b, "negatives": v_neg}

    # --- held-out eval sets (untouched originals) ---
    td_n, td_b, td_neg = add_dut(dut_yolo, "test", dirs[("images", "test_dut")],
                                 dirs[("labels", "test_dut")], mapping, "test_dut")
    tm_n, tm_b, tm_neg = add_maciullo(merged, maci, "test", dirs[("images", "test_maciullo")],
                                      dirs[("labels", "test_maciullo")], 1, mapping)
    stats["test_dut"] = {"images": td_n, "boxes": td_b, "negatives": td_neg}
    stats["test_maciullo"] = {"images": tm_n, "boxes": tm_b, "negatives": tm_neg}

    # --- data.yaml for training + eval yamls (Phase 4) ---
    cfg = Path(a.repo) / "configs"
    (cfg / "merged_drone.yaml").write_text(
        f"# Merged DUT + Maciullo (leakage-safe). Generated by scripts/merge_datasets.py\n"
        f"path: {merged}\ntrain: images/train\nval: images/val\ntest: images/test_dut\n\n"
        f"names:\n  0: drone\n")
    for name, testdir in (("test_dut", "images/test_dut"),
                          ("test_maciullo", "images/test_maciullo")):
        (cfg / f"eval_{name}.yaml").write_text(
            f"# Held-out eval set '{name}'. val: points to it so `--split val` evaluates it.\n"
            f"path: {merged}\ntrain: images/train\nval: {testdir}\ntest: {testdir}\n\n"
            f"names:\n  0: drone\n")

    # --- reproducibility: full image->split mapping dump ---
    rep = Path(a.repo) / "reports"
    rep.mkdir(exist_ok=True)
    with (rep / "split_mapping.csv").open("w") as f:
        f.write("source,file,dest_split,n_boxes\n")
        for src, file, dest, nb in mapping:
            f.write(f"{src},{file},{dest},{nb}\n")
    (rep / "merge_stats.json").write_text(json.dumps(stats, indent=2))

    print("=== MERGE STATS ===")
    print(json.dumps(stats, indent=2))
    print(f"\nsplit mapping -> {rep/'split_mapping.csv'} ({len(mapping)} rows)")
    print(f"data.yaml     -> {cfg/'merged_drone.yaml'}")


if __name__ == "__main__":
    main()
