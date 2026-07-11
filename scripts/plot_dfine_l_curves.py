#!/usr/bin/env python3
"""D-FINE-L@960 training curves + reference AP comparison vs yolo26l / D-FINE-N.

Input : reports/dfine_l960_metrics.json  (merged-val, COCO eval, 120ep)
Output: reports/dfine_l960_curves.png

WARNING — eval sets differ:
  - D-FINE-L@960 AP = merged-val (training val split), COCO eval.
  - yolo26l-P2(H) / D-FINE-N = held-out test (DUT / Maciullo).
  The two axes are different data -> absolute heights are NOT directly comparable;
  read trend / rough position only. For a same-test comparison run
  scripts/dfine_eval.py on best_stg2.pth.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
METRICS = ROOT / "reports/dfine_l960_metrics.json"
OUT = ROOT / "reports/dfine_l960_curves.png"
STOP_EPOCH = 108  # stg1 -> stg2 (aug off + EMA restart)

d = json.load(open(METRICS))
m = d["metrics"]
ep = [r["epoch"] for r in m]
best_ep, best_ap = d["best_epoch"], d["best_AP"]

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))

# ---- Left: training curves (merged-val) ----
for key, label, color in [
    ("AP50", "AP50", "#4C9F70"),
    ("AP75", "AP75", "#E1A730"),
    ("AP", "AP@[.50:.95]", "#3B6FB0"),
    ("AR100", "AR@100", "#9B59B6"),
]:
    axL.plot(ep, [r[key] for r in m], label=label, color=color, lw=1.8)
axL.axvline(STOP_EPOCH, ls="--", color="#888", lw=1)
axL.text(STOP_EPOCH - 1.5, 0.905, "stop_epoch 108\n(aug off / EMA restart)",
         fontsize=8, color="#555", va="top", ha="right")
axL.scatter([best_ep], [best_ap], color="#C0392B", zorder=5, s=45)
axL.annotate(f"best {best_ap:.4f}\n@ep{best_ep} (stg2)",
             (best_ep, best_ap), textcoords="offset points", xytext=(-72, -6),
             fontsize=8, color="#C0392B")
axL.set_xlabel("epoch")
axL.set_ylabel("AP / AR  (COCO eval, merged-val)")
axL.set_title("D-FINE-L@960 training curves (merged-val, 120ep)")
axL.set_ylim(0.58, 1.0)
axL.grid(alpha=0.25)
axL.legend(loc="lower right", fontsize=8)

# ---- Right: final AP50-95 (eval sets differ -> grouped & separated) ----
groups = [
    ("DUT\n(test)", [("yolo26l-P2", 0.769, "#B0563B"), ("D-FINE-N", 0.705, "#7FB0A0")]),
    ("Maciullo\n(test)", [("yolo26l-P2", 0.450, "#B0563B"), ("D-FINE-N", 0.428, "#7FB0A0")]),
    ("merged-val\n(!= test)", [("D-FINE-L@960", best_ap, "#3B6FB0")]),
]
xpos, xticks, xlabels = 0.0, [], []
seen = set()
for gname, bars in groups:
    gcenter = xpos + (len(bars) - 1) * 0.5 * 0.9
    for name, val, color in bars:
        lbl = name if name not in seen else None
        seen.add(name)
        axR.bar(xpos, val, width=0.8, color=color, label=lbl)
        axR.text(xpos, val + 0.008, f"{val:.3f}", ha="center", fontsize=8)
        xpos += 0.9
    xticks.append(gcenter)
    xlabels.append(gname)
    xpos += 0.7
axR.axvspan(xticks[-1] - 0.9, xpos - 0.7, color="#F2C94C", alpha=0.12)
axR.set_xticks(xticks)
axR.set_xticklabels(xlabels, fontsize=8)
axR.set_ylabel("AP@[.50:.95]")
axR.set_ylim(0, 1.0)
axR.set_title("Final AP50-95  (WARNING: eval sets differ, not directly comparable)")
axR.grid(axis="y", alpha=0.25)
axR.legend(loc="upper right", fontsize=8)
axR.text(0.5, -0.16,
         "D-FINE-L = merged-val COCO eval; others = held-out test. "
         "Same-test comparison needs dfine_eval.py.",
         transform=axR.transAxes, ha="center", fontsize=7.5, color="#777")

fig.suptitle("D-FINE-L@960  (merged_drone / 3xRTX4090 / 120ep) — convergence & reference comparison",
             fontsize=12)
fig.tight_layout(rect=[0, 0.02, 1, 0.97])
fig.savefig(OUT, dpi=130)
print(f"saved: {OUT}")
