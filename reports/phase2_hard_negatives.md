# Phase 2 — Hard negatives (skip 기록)

- NEG_DIR 미지정 → **skip**.
- Maciullo는 사실상 all-positive (train 51,446장 중 무객체 1장). 병합 train 전체 negative 4장(0.007%).
- 결론: **병합만으로 배경 clutter FP는 구조적으로 안 줄어든다** — 라벨-없는 배경(negative) 셋을 NEG_DIR로 공급해야 함 (train의 ≤15% cap, recall 저하 방지).
- Phase 4에서 관측된 FP 개선은 데이터 다양성의 부수 효과로 해석할 것.
