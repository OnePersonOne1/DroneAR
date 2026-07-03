# Old (DUT-only) vs New (merged) — Phase 4

## 데이터셋 확장
지면-드론(Maciullo) 데이터를 DUT(공중)에 leakage-safe 병합 → 공중뿐 아니라 지면과 겹치는 드론의 탐지 성능을 올렸다.

## 결과 (고정 held-out test) — 3-way

| 모델 | test set | mAP@0.5 | mAP@0.5:0.95 | P | R | FP/img | small-recall(<32px) |
|---|---|---:|---:|---:|---:|---:|---:|
| old (DUT-only, 150ep) | DUT-test | 0.951 | 0.648 | 0.963 | 0.922 | 0.063 | 0.932 |
| merged-100ep | DUT-test | 0.927 | 0.619 | 0.953 | 0.870 | 0.063 | 0.842 |
| **merged-300ep** | DUT-test | 0.950 | **0.650** | **0.971** | 0.922 | **0.041** | 0.889 |
| old (DUT-only, 150ep) | Maciullo-test | 0.601 | 0.216 | 0.776 | 0.569 | 0.213 | 0.617 |
| merged-100ep | Maciullo-test | **0.891** | **0.445** | 0.924 | **0.836** | **0.047** | **0.829** |
| **merged-300ep** | Maciullo-test | 0.858 | 0.415 | 0.916 | 0.822 | 0.067 | 0.798 |

- 300ep: DUT 회복(≈old 동급, FP/img 최저) + Maciullo 대폭 개선 유지 → **균형 최적**.
- 100ep: Maciullo 최고, DUT 소폭 희생 → 지면 특화.
- best.pt 선택 기준이 DUT-val이라 장기 학습(300ep)이 DUT 쪽으로 재수렴.

> 비교 조건: best.pt vs best.pt, 동일 test set. 이미지 노출량 old 0.70M / 100ep 5.66M / 300ep ~17M — epoch cap이 merged를 불리하게 하지 않음.

## Failure-mode detail
```
{
  "old_DUT_only_150ep": {
    "DUT-test":      {"images": 2200, "TP": 2095, "FP": 138, "FP_per_image": 0.0627,
                      "recall_small_<32px": 0.9319, "recall_medium": 0.9387, "recall_large_>96px": 0.9304},
    "Maciullo-test": {"images": 2625, "TP": 1619, "FP": 559, "FP_per_image": 0.213,
                      "recall_small_<32px": 0.6169, "recall_medium": 0.6418, "recall_large_>96px": 0.4199}
  },
  "merged_100ep": {
    "DUT-test":      {"images": 2200, "TP": 1979, "FP": 139, "FP_per_image": 0.0632,
                      "recall_small_<32px": 0.842,  "recall_medium": 0.9328, "recall_large_>96px": 0.961},
    "Maciullo-test": {"images": 2625, "TP": 2385, "FP": 122, "FP_per_image": 0.0465,
                      "recall_small_<32px": 0.829,  "recall_medium": 0.7655, "recall_large_>96px": 0.921}
  },
  "merged_300ep": {
    "DUT-test":      {"images": 2200, "TP": 2066, "FP": 89,  "FP_per_image": 0.0405,
                      "recall_small_<32px": 0.8891, "recall_medium": 0.9644, "recall_large_>96px": 0.9777},
    "Maciullo-test": {"images": 2625, "TP": 2313, "FP": 175, "FP_per_image": 0.0667,
                      "recall_small_<32px": 0.7982, "recall_medium": 0.73,   "recall_large_>96px": 0.9142}
  }
}
```

(GT-empty 이미지는 두 test set 모두 0장 → FP_on_empty=0. n_small/med/large: DUT-test 1380/506/359, Maciullo-test 877/1100/886.)
