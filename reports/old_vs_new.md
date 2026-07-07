# old vs new 평가 (생성물)

> `scripts/eval_compare.py` 산출물 — 재실행 시 덮어써짐. **정확도 비교의 단일 출처는
> [ablation_matrix.md](ablation_matrix.md)** (이 파일은 원시 결과 보관용).

마지막 실행: old=DUT-only 150ep / new=merged (100ep·300ep), imgsz 640, 고정 held-out test.

## Failure-mode detail (원시)
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
(GT-empty 이미지 0장 → FP_on_empty=0. n_small/med/large: DUT-test 1380/506/359, Maciullo-test 877/1100/886.)

- 100ep vs 300ep는 Pareto 관계(이미지-가중 mAP50 0.907 vs 0.900) — 배포 도메인 prior로 선택.
- 비교 조건: best.pt vs best.pt, 동일 test set. 이미지 노출량 old 0.70M / 100ep 5.66M / 300ep ~17M.
