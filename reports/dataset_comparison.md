# Dataset comparison — DUT vs Maciullo vs merged (train split)

## bbox size distribution (from YOLO labels)

| set | boxes | side@640 p50 | p5..p95 | small<32px | med | large>96px |
|---|---:|---:|---|---:|---:|---:|
| DUT_train | 5243 | 13.9 | 6.65..145.41 | 0.8114 | 0.1024 | 0.0862 |
| Maciullo_train | 52676 | 46.16 | 13.27..362.66 | 0.3462 | 0.3872 | 0.2666 |
| merged_train | 57919 | 42.33 | 10.95..354.72 | 0.3883 | 0.3614 | 0.2503 |

## background composition proxy (Sobel edge density; sky=low, ground/clutter=high)

sample=800/set, ground-like cut = 12.0

| set | sampled | edge dens p50 | mean | frac ground-like |
|---|---:|---:|---:|---:|
| DUT_train | 800 | 33.36 | 39.76 | 0.9738 |
| Maciullo_train | 800 | 35.61 | 38.34 | 0.8838 |
| merged_train | 800 | 35.03 | 37.99 | 0.88 |

## reading
- If Maciullo shows **higher edge density / higher frac ground-like** than DUT,
  it is adding non-sky (ground/terrain/urban) backgrounds — the intended lever
  for reducing ground-background misses and ground-clutter false positives.
- bbox-size overlap tells whether the merge shifts the object-scale regime
  (relevant to small-object recall in Phase 4).
- Proxy caveat: edge density is computed over the whole frame (drone + bg);
  it is a coarse sky-vs-ground signal, not a segmentation.
