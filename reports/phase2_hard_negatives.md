# Phase 2 — Hard negatives

NEG_DIR = "" (unset) -> **skipped**.

## Warning (explicit)
The Maciullo dataset is effectively all-positive: of 51,446 train images only **1**
has zero drone boxes (0.00%). The merged train split contains just **4** negative
(background-only) images total (DUT 3 + Maciullo 1) = 0.007%.

Consequence: **the merge alone does not structurally reduce ground-clutter false
positives.** To actually suppress background FP on ground/terrain/urban scenes, supply
label-free hard-negative background images via NEG_DIR (recommended cap: <=15% of train
to avoid recall degradation). No negatives were added in this run, so any background-FP
improvement observed in Phase 4 is an incidental effect of data/scale diversity, not of
targeted hard-negative mining.
