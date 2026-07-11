#!/usr/bin/env python3
"""reports/unified/*.json → README용 마스터 정확도표(md).

전 모델 동일 조건: held-out test(DUT·Maciullo) · faster-coco-eval · conf 0.25 · IoU-match 0.5.
지표 보존: 표엔 AP50/AP50-95 · far(<16px) · <8px · FP/img. 전체 size-bin은 json에 보존.
정렬: 계열·크기순(yolo26n→s→l, P2, D-FINE-N→L).

usage: python scripts/build_master_table.py  [--out reports/master_table.md]
"""
import argparse
import json
from pathlib import Path

# (key, 표시명, train, imgsz, ablation ref) — 이 순서가 곧 표 정렬순
SPEC = [
    ("yolo26n_640_dut",    "yolo26n",     "DUT",    640, "A"),
    ("yolo26n_960_dut",    "yolo26n",     "DUT",    960, "B"),
    ("yolo26s_640",        "yolo26s",     "DUT",    640, "—"),
    ("yolo26s_960",        "yolo26s",     "DUT",    960, "—"),
    ("yolo26n_640_m100",   "yolo26n",     "merged", 640, "C"),
    ("yolo26n_640_m300",   "yolo26n",     "merged", 640, "D"),
    ("yolo26nP2_960_m100", "yolo26n-P2",  "merged", 960, "E/F"),
    ("yolo26lP2_960_m100", "yolo26l-P2",  "merged", 960, "H/I"),
    ("dfine_n_640_m220",   "D-FINE-N",    "merged", 640, "—"),
    ("dfine_l_960_m120",   "D-FINE-L",    "merged", 960, "—"),
]

HDR = ("| 모델 | ref | train | imgsz | DUT AP50 | DUT AP50-95 | DUT far(<16px) | DUT <8px | DUT FP/img "
       "| Maci AP50 | Maci AP50-95 | Maci far(<16px) | Maci <8px | Maci FP/img |")
SEP = "|---|:--:|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"


def fmt(x, nd=3):
    return "—" if x is None else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unified-dir", default="reports/unified")
    ap.add_argument("--out", default="reports/master_table.md")
    a = ap.parse_args()
    udir = Path(a.unified_dir)

    rows, missing = [], []
    # 계열별 최고 AP50(각 test) 볼드 처리용
    best = {"dut": -1, "maci": -1}
    data = {}
    for key, name, train, imgsz, ref in SPEC:
        f = udir / f"{key}.json"
        if not f.exists():
            missing.append(key); continue
        j = json.loads(f.read_text())
        data[key] = j
        best["dut"] = max(best["dut"], j["test_dut"]["AP50"])
        best["maci"] = max(best["maci"], j["test_maciullo"]["AP50"])

    for key, name, train, imgsz, ref in SPEC:
        if key not in data:
            continue
        j = data[key]
        d, m = j["test_dut"], j["test_maciullo"]

        def cell(v, is_best, nd=3):
            s = fmt(v, nd)
            return f"**{s}**" if (is_best and v is not None) else s

        rows.append(
            f"| {name} | {ref} | {train} | {imgsz} "
            f"| {cell(d['AP50'], d['AP50'] == best['dut'])} | {fmt(d['AP50_95'])} "
            f"| {fmt(d['far_recall'])} | {fmt(d['recall_by_bin']['<8'])} | {fmt(d['FP_per_image'])} "
            f"| {cell(m['AP50'], m['AP50'] == best['maci'])} | {fmt(m['AP50_95'])} "
            f"| {fmt(m['far_recall'])} | {fmt(m['recall_by_bin']['<8'])} | {fmt(m['FP_per_image'])} |"
        )

    note = ("> 전 모델 **동일 조건**: held-out test(DUT-test 2200장·Maciullo-test 2625장) · "
            "**faster-coco-eval** · **conf 0.25 · IoU-match 0.5** · side@640 size-bin. "
            "far = <16px(=<8+8-16) recall. AP는 conf 0.25 운영점 필터 COCO(=ultralytics val 저conf 표준과 산출기 다름 → 상단 old 표보다 낮게 측정됨, 대신 전 모델 상호 비교 가능). "
            "전체 size-bin recall은 `reports/unified/<key>.json`에 보존.")

    md = ["# 정확도 마스터표 (생성물 — `scripts/build_master_table.py`)", "",
          "> `reports/unified/*.json` 단일 출처. 재생성 시 덮어써짐.", "",
          HDR, SEP, *rows, "", note, ""]
    if missing:
        md.append(f"> ⚠️ 미측정(json 없음): {', '.join(missing)}")
    Path(a.out).write_text("\n".join(md) + "\n")
    print(f"saved {a.out}  ({len(rows)}/{len(SPEC)} models)")
    if missing:
        print("missing:", missing)
    print("\n".join([HDR, SEP, *rows]))


if __name__ == "__main__":
    main()
