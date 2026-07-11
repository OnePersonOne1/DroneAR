#!/usr/bin/env python3
"""정확도-속도 트레이드오프 — AP vs latency (전 모델 통일 벤치).

x = latency(ms, 낮을수록 빠름), y = DUT AP50-95(COCO). 좌=GPU(4090 fp32)·우=CPU(Ryzen t8).
점 = 배포 대표 모델(merged yolo26n/s/l-P2 · D-FINE-N/L). 색 = 계열(yolo/D-FINE).
원자료: reports/unified_latency.json(속도) · reports/unified/*.json(AP).
산출: reports/ap_latency.png. usage: python scripts/plot_ap_latency.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 벤치 key → (표시명, unified AP key, family)
MAP = [
    ("yolo26n_m300", "yolo26n·640",    "yolo26n_640_m300",   "yolo"),
    ("yolo26s_640",  "yolo26s·640",    "yolo26s_640",        "yolo"),
    ("yolo26lP2",    "yolo26l-P2·960", "yolo26lP2_960_m100", "yolo"),
    ("dfine_n",      "D-FINE-N·640",   "dfine_n_640_m220",   "dfine"),
    ("dfine_l",      "D-FINE-L·960",   "dfine_l_960_m120",   "dfine"),
]
BLUE, ORANGE = "#2a78d6", "#eb6834"       # yolo, D-FINE (검증 슬롯)
FCOL = {"yolo": BLUE, "dfine": ORANGE}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e2"


def main():
    lat = json.loads(Path("reports/unified_latency.json").read_text())
    udir = Path("reports/unified")
    pts = []
    for key, name, apkey, fam in MAP:
        ap = json.loads((udir / f"{apkey}.json").read_text())["test_dut"]["AP50_95"]
        L = lat[key]
        pts.append(dict(name=name, fam=fam, ap=ap,
                        gpu=L.get("gpu_fp32", {}).get("ms"),
                        cpu=L.get("cpu_t8", {}).get("ms")))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4))
    fig.patch.set_facecolor("white")
    panels = [("gpu", "GPU latency — RTX 4090 (fp32, ms)", axes[0]),
              ("cpu", "CPU latency — Ryzen 9 7950X (8 threads, ms)", axes[1])]

    for field, title, ax in panels:
        ax.set_facecolor("white")
        xs = [p[field] for p in pts]
        for p in pts:
            ax.scatter(p[field], p["ap"], s=120, color=FCOL[p["fam"]],
                       edgecolor="white", linewidth=1.4, zorder=4)
            ax.annotate(f"{p['name']}\nAP50-95 {p['ap']:.3f} · {p[field]:.1f}ms",
                        (p[field], p["ap"]), xytext=(6, 6), textcoords="offset points",
                        fontsize=8, color=INK, zorder=5)
        ax.set_xlim(left=0, right=max(xs) * 1.28)
        ax.set_ylim(0.55, 0.82)
        ax.set_xlabel(title, fontsize=10, color=MUTED)
        ax.set_title(title.split(" — ")[0], fontsize=11.5, color=INK, fontweight="bold", pad=8)
        ax.grid(True, color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(length=0, labelsize=9, colors=MUTED)
    axes[0].set_ylabel("DUT-test AP50-95 (COCO)", fontsize=10, color=MUTED)

    # 범례(계열)
    handles = [plt.Line2D([], [], marker="o", ls="", ms=10, color=BLUE, label="yolo26"),
               plt.Line2D([], [], marker="o", ls="", ms=10, color=ORANGE, label="D-FINE")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, 0.965))
    fig.suptitle("Accuracy vs latency — deployment imgsz, pure forward, batch 1",
                 fontsize=12.5, color=INK, y=0.995, fontweight="bold")
    fig.text(0.5, 0.005, "Left-up = better (higher AP, lower latency). All fp32 (unified precision). "
             "D-FINE-L = highest AP but slowest; D-FINE-N is GPU-efficient (195 FPS); yolo26n = fastest.",
             ha="center", fontsize=8.0, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.90))
    out = Path("reports/ap_latency.png")
    fig.savefig(out, dpi=150, facecolor="white")
    print("saved", out)


if __name__ == "__main__":
    main()
