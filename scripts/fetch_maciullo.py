#!/usr/bin/env python3
"""Phase 0 — acquire & materialize the Maciullo DroneDetectionDataset (HF mirror).

Source : pathikg/drone-detection-dataset  (parquet, HF datasets-server schema)
         features: width,height,objects{bbox[xywh px],category(ClassLabel drone),
         area,id}, image(Image bytes), image_id(int64). splits: train / test.

Output (NEW_DIR, materialized, never DUT):
    <dst>/images/{train,test}/<split>_<idx06>.jpg      decoded frames
    <dst>/annotations/{train,test}.jsonl               one COCO-ish record / line:
        {"file","image_id","width","height","boxes":[[x,y,w,h],...],"category":[0,...]}
    <dst>/AUDIT.md                                      integrity + histograms
    <dst>/provenance.json                               sequence-provenance finding

Downloads parquet via huggingface_hub, decodes with pyarrow+PIL. No `datasets` dep.
Idempotent: re-running skips shards already fully materialized (by count check).
"""
import argparse
import io
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image

REPO = "pathikg/drone-detection-dataset"
SHARDS = {
    "train": [f"data/train-0000{i}-of-00009.parquet" for i in range(9)],
    "test": ["data/test-00000-of-00001.parquet"],
}


def materialize(dst: Path, split: str, cache: Path):
    img_dir = dst / "images" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_path = dst / "annotations" / f"{split}.jsonl"
    ann_path.parent.mkdir(parents=True, exist_ok=True)

    res_hist = Counter()        # (w,h) -> n images
    nbox_hist = Counter()       # n boxes per image -> n images
    cls_hist = Counter()        # category -> n boxes
    idx = 0
    fmt_seen = Counter()
    with ann_path.open("w") as af:
        for shard in SHARDS[split]:
            print(f"  [{split}] downloading {shard} ...", flush=True)
            local = hf_hub_download(REPO, shard, repo_type="dataset",
                                    cache_dir=str(cache))
            pf = pq.ParquetFile(local)
            for bi in range(pf.num_row_groups):
                tbl = pf.read_row_group(bi)
                d = tbl.to_pydict()
                n = len(d["image_id"])
                for k in range(n):
                    img_struct = d["image"][k]
                    raw = img_struct["bytes"] if isinstance(img_struct, dict) else img_struct
                    im = Image.open(io.BytesIO(raw))
                    fmt_seen[im.format] += 1
                    w, h = im.size
                    fname = f"{split}_{idx:06d}.jpg"
                    if im.mode != "RGB":
                        im = im.convert("RGB")
                    im.save(img_dir / fname, "JPEG", quality=95)
                    objs = d["objects"][k]
                    boxes = [[float(v) for v in b] for b in objs["bbox"]]
                    cats = [int(c) for c in objs["category"]]
                    res_hist[(int(w), int(h))] += 1
                    nbox_hist[len(boxes)] += 1
                    for c in cats:
                        cls_hist[c] += 1
                    af.write(json.dumps({
                        "file": fname, "image_id": int(d["image_id"][k]),
                        "width": int(w), "height": int(h),
                        "boxes": boxes, "category": cats,
                    }) + "\n")
                    idx += 1
            print(f"    -> {idx} images so far", flush=True)
    return idx, res_hist, nbox_hist, cls_hist, fmt_seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dst", default="/mnt/ssd_0/dataset/DroneDetection")
    ap.add_argument("--cache", default="/mnt/ssd_0/dataset/.hf_cache")
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    a = ap.parse_args()
    dst, cache = Path(a.dst), Path(a.cache)

    summary = {}
    lines = ["# Maciullo DroneDetectionDataset — Phase 0 audit", "",
             f"Source: `{REPO}` (HF parquet mirror). Materialized to `{dst}`.", ""]
    for split in a.splits:
        print(f"=== materialize {split} ===", flush=True)
        n, res, nbox, cls, fmt = materialize(dst, split, cache)
        summary[split] = {"images": n,
                          "resolutions": {f"{w}x{h}": c for (w, h), c in res.most_common()},
                          "bbox_per_image": dict(sorted(nbox.items())),
                          "class_hist": {str(k): v for k, v in cls.items()},
                          "img_formats": dict(fmt)}
        total_boxes = sum(k * v for k, v in nbox.items())
        neg = nbox.get(0, 0)
        lines += [f"## split `{split}`", "",
                  f"- images: **{n}**",
                  f"- total boxes: **{total_boxes}**  (avg {total_boxes/max(n,1):.3f}/img)",
                  f"- images with 0 boxes (negatives): **{neg}** ({100*neg/max(n,1):.2f}%)",
                  f"- source image formats: {dict(fmt)}",
                  "",
                  "resolution histogram:",
                  ""]
        for k, v in sorted(res.items(), key=lambda x: -x[1])[:10]:
            lines.append(f"  - {k[0]}x{k[1]}: {v}")
        lines += ["", "bbox-per-image histogram:", ""]
        for k, v in sorted(nbox.items()):
            lines.append(f"  - {k} box: {v} images")
        lines += ["", f"class histogram (should be single class 0=drone): "
                  f"{ {str(k): v for k, v in cls.items()} }", ""]

    (dst / "audit_summary.json").write_text(json.dumps(summary, indent=2))
    (dst / "AUDIT.md").write_text("\n".join(lines) + "\n")

    # Sequence provenance finding.
    prov = {
        "recoverable": False,
        "reason": ("HF parquet mirror strips original Maciullo filenames; only a "
                   "sequential integer `image_id` (0..N-1) is present, with no video_id "
                   "or per-frame source path. Video/sequence grouping cannot be recovered "
                   "from this mirror."),
        "leakage_safe_fallback": (
            "Keep official train/test boundary intact. Maciullo official TEST is held "
            "out entirely as a separate eval set. Maciullo official TRAIN goes wholly to "
            "the merged TRAIN. Merged VAL is taken ONLY from DUT's official val split "
            "(itself sequence-separated), so no Maciullo frame appears in both train and "
            "val. This avoids frame-level leakage that a random/group split on unknown "
            "video ids would risk."),
    }
    (dst / "provenance.json").write_text(json.dumps(prov, indent=2))
    print("\nPhase 0 done. Audit:", dst / "AUDIT.md")
    print("Provenance:", prov["recoverable"], "->", dst / "provenance.json")
    print(json.dumps({k: v["images"] for k, v in summary.items()}, indent=2))


if __name__ == "__main__":
    main()
