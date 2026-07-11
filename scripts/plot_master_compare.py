#!/usr/bin/env python3
"""reports/unified/*.json → 마스터표 정면 비교 차트(PNG).

전 모델 동일 조건(held-out test·faster-coco-eval·conf0.25/IoU0.5)이라 막대 높이 직접 비교 가능.
시리즈 = 도메인(DUT / Maciullo), 패널 = 지표(AP50 · AP50-95 · far<16px). 색은 dataviz 레퍼런스
검증 슬롯(blue #2a78d6 · orange #eb6834, CVD-안전). 산출: reports/master_compare.png.

usage: python scripts/plot_master_compare.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (key, 표시명, train, imgsz) — 위→아래 표시순(마스터표와 동일 계열순)
SPEC = [
    ("yolo26n_640_dut",    "yolo26n",    "DUT",    640),
    ("yolo26n_960_dut",    "yolo26n",    "DUT",    960),
    ("yolo26s_640",        "yolo26s",    "DUT",    640),
    ("yolo26s_960",        "yolo26s",    "DUT",    960),
    ("yolo26n_640_m100",   "yolo26n",    "merged", 640),
    ("yolo26n_640_m300",   "yolo26n",    "merged", 640),
    ("yolo26nP2_960_m100", "yolo26n-P2", "merged", 960),
    ("yolo26lP2_960_m100", "yolo26l-P2", "merged", 960),
    ("dfine_n_640_m220",   "D-FINE-N",   "merged", 640),
    ("dfine_l_960_m120",   "D-FINE-L",   "merged", 960),
]

BLUE, ORANGE = "#2a78d6", "#eb6834"          # DUT, Maciullo (검증 슬롯)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e2"
PANELS = [("AP50", "AP50"), ("AP50_95", "AP50-95"), ("far_recall", "far-recall (<16px)")]


def main():
    udir = Path("reports/unified")
    rows = []
    for key, name, train, imgsz in SPEC:
        j = json.loads((udir / f"{key}.json").read_text())
        rows.append((f"{name}·{imgsz}  {train}", j["test_dut"], j["test_maciullo"]))
    labels = [r[0] for r in rows]
    n = len(rows)
    y = list(range(n))
    h = 0.38

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 6.4), sharey=True)
    fig.patch.set_facecolor("white")

    for ax, (mkey, mlabel) in zip(axes, PANELS):
        ax.set_facecolor("white")
        dut = [r[1][mkey] for r in rows]
        mac = [r[2][mkey] for r in rows]
        yb = [(n - 1 - i) for i in y]          # 첫 모델을 위로
        bars_d = ax.barh([v + h/2 for v in yb], dut, height=h, color=BLUE,
                         label="DUT-test", zorder=3)
        bars_m = ax.barh([v - h/2 for v in yb], mac, height=h, color=ORANGE,
                         label="Maciullo-test", zorder=3)
        # 값 라벨(선택적, 막대 끝)
        for bars, vals in ((bars_d, dut), (bars_m, mac)):
            for b, v in zip(bars, vals):
                ax.text(v + 0.012, b.get_y() + b.get_height()/2, f"{v:.2f}",
                        va="center", ha="left", fontsize=7.2, color=MUTED)
        ax.set_xlim(0, 1.06)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_title(mlabel, fontsize=11.5, color=INK, pad=8, fontweight="bold")
        ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(length=0, labelsize=8.5, colors=MUTED)

    axes[0].set_yticks([(n - 1 - i) for i in y])
    axes[0].set_yticklabels(labels, fontsize=8.6, color=INK)
    axes[0].tick_params(axis="y", labelcolor=INK)

    # 범례(2 시리즈 → 항상 표시) — figure 상단 가로 배치(막대와 겹침 방지)
    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE),
               plt.Rectangle((0, 0), 1, 1, color=ORANGE)]
    fig.legend(handles, ["DUT-test", "Maciullo-test"], loc="upper center",
               ncol=2, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 0.945),
               handlelength=1.1, columnspacing=1.6)

    fig.suptitle("All-model comparison — held-out test · faster-coco-eval · conf 0.25 / IoU 0.5",
                 fontsize=12.5, color=INK, y=0.995, fontweight="bold")
    fig.text(0.5, 0.008,
             "Identical-condition unified eval (reports/unified/*.json). "
             "D-FINE-L tops both domains on every metric. AP = conf-0.25 operating-point COCO.",
             ha="center", fontsize=8.2, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.90))
    out = Path("reports/master_compare.png")
    fig.savefig(out, dpi=150, facecolor="white")
    print("saved", out)


if __name__ == "__main__":
    main()
