# Old (DUT-only) vs New (merged) — Phase 4

## 데이터셋 확장
지면-드론(Maciullo) 데이터를 DUT(공중)에 leakage-safe 병합 → 공중뿐 아니라 지면과 겹치는 드론의 탐지 성능을 올렸다.

## 결과 (고정 held-out test)

| 모델 | test set | mAP@0.5 | mAP@0.5:0.95 | P | R | FP/img | small-recall(<32px) |
|---|---|---:|---:|---:|---:|---:|---:|
| old (DUT-only) | DUT-test | 0.951 | 0.648 | 0.963 | 0.922 | 0.063 | 0.932 |
| new (merged) | DUT-test | 0.927 | 0.619 | 0.953 | 0.870 | 0.063 | 0.842 |
| old (DUT-only) | Maciullo-test | 0.601 | 0.216 | 0.776 | 0.569 | 0.213 | 0.617 |
| new (merged) | Maciullo-test | 0.891 | 0.445 | 0.924 | 0.836 | 0.047 | 0.829 |

> 비교 조건: best.pt vs best.pt, 동일 test set. old 150ep(best@134) / new 100ep — 이미지 노출량은 new(5.66M)가 old(0.70M)보다 많아 epoch cap이 new를 불리하게 하지 않음.

## Failure-mode detail
```
{
  "old_DUT_only": {
    "DUT-test": {
      "images": 2200,
      "TP": 2095,
      "FP": 138,
      "FP_per_image": 0.0627,
      "GT_empty_images": 0,
      "FP_on_empty_images": 0,
      "recall_small_<32.0px": 0.9319,
      "n_small": 1380,
      "recall_medium": 0.9387,
      "n_medium": 506,
      "recall_large_>96.0px": 0.9304,
      "n_large": 359
    },
    "Maciullo-test": {
      "images": 2625,
      "TP": 1619,
      "FP": 559,
      "FP_per_image": 0.213,
      "GT_empty_images": 0,
      "FP_on_empty_images": 0,
      "recall_small_<32.0px": 0.6169,
      "n_small": 877,
      "recall_medium": 0.6418,
      "n_medium": 1100,
      "recall_large_>96.0px": 0.4199,
      "n_large": 886
    }
  },
  "new_merged": {
    "DUT-test": {
      "images": 2200,
      "TP": 1979,
      "FP": 139,
      "FP_per_image": 0.0632,
      "GT_empty_images": 0,
      "FP_on_empty_images": 0,
      "recall_small_<32.0px": 0.842,
      "n_small": 1380,
      "recall_medium": 0.9328,
      "n_medium": 506,
      "recall_large_>96.0px": 0.961,
      "n_large": 359
    },
    "Maciullo-test": {
      "images": 2625,
      "TP": 2385,
      "FP": 122,
      "FP_per_image": 0.0465,
      "GT_empty_images": 0,
      "FP_on_empty_images": 0,
      "recall_small_<32.0px": 0.829,
      "n_small": 877,
      "recall_medium": 0.7655,
      "n_medium": 1100,
      "recall_large_>96.0px": 0.921,
      "n_large": 886
    }
  }
}
```
