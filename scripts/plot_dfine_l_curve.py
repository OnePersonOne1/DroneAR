#!/usr/bin/env python3
"""D-FINE-L 학습 곡선 — DUT-val COCO AP(mAP50·mAP50-95) vs epoch.

원자료: reports/dfine_l960_train_log.txt (JSONL, epoch당 test_coco_eval_bbox[0]=mAP50-95, [1]=mAP50).
stg1/stg2 경계(ep108) · best(ep115) 표시. 산출: reports/dfine_l960_ap_curve.png.

usage: python scripts/plot_dfine_l_curve.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE = "#2a78d6", "#eb6834"          # mAP50, mAP50-95 (검증 슬롯)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e2"
STG2_START, BEST_EP = 108, 115


def main():
    lines = Path("reports/dfine_l960_train_log.txt").read_text().splitlines()
    ep, map50, map5095 = [], [], []
    for i, ln in enumerate(lines):
        j = json.loads(ln)
        c = j.get("test_coco_eval_bbox")
        if not c:
            continue
        ep.append(i)
        map5095.append(c[0])
        map50.append(c[1])

    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.plot(ep, map50, color=BLUE, lw=2, label="mAP50", zorder=3)
    ax.plot(ep, map5095, color=ORANGE, lw=2, label="mAP50-95", zorder=3)

    # stg1→stg2 경계
    ax.axvline(STG2_START, color=MUTED, lw=1, ls="--", zorder=2)
    ax.text(STG2_START - 1.5, 0.06, "stg1  |  stg2 (aug off, EMA restart)",
            rotation=90, va="bottom", ha="right", fontsize=8, color=MUTED)
    # best 마커
    bi = ep.index(BEST_EP) if BEST_EP in ep else None
    if bi is not None:
        for series, col in ((map50, BLUE), (map5095, ORANGE)):
            ax.scatter([BEST_EP], [series[bi]], color=col, s=42, zorder=5,
                       edgecolor="white", linewidth=1.2)
        ax.annotate(f"best ep{BEST_EP}\nmAP50 {map50[bi]:.3f} · mAP50-95 {map5095[bi]:.3f}",
                    (BEST_EP, map5095[bi]), xytext=(BEST_EP - 30, 0.40),
                    fontsize=8.5, color=INK,
                    arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))

    ax.set_xlim(0, len(lines) - 1)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("epoch", fontsize=10, color=MUTED)
    ax.set_ylabel("COCO AP (DUT-val)", fontsize=10, color=MUTED)
    ax.set_title("D-FINE-L@960 training curve — DUT-val COCO AP per epoch",
                 fontsize=12, color=INK, fontweight="bold", pad=8)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(length=0, labelsize=9, colors=MUTED)
    ax.legend(loc="lower right", frameon=False, fontsize=10)

    fig.tight_layout()
    out = Path("reports/dfine_l960_ap_curve.png")
    fig.savefig(out, dpi=150, facecolor="white")
    print("saved", out, "| best ep", BEST_EP, "mAP50-95", round(map5095[bi], 4) if bi else None)


if __name__ == "__main__":
    main()
