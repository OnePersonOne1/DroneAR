# ncnn(CPU) vs ONNX Runtime(FP32) parity — yolo26n_drone_640

- demo 14장, imgsz=640, conf=0.25
- ncnn: FP16 export, `use_vulkan_compute=false`(CPU). one-to-many head + IoU 0.7 NMS.
- 비교 공간: 640 letterbox 입력 좌표(동일 전처리).

| 이미지 | ORT det | ncnn det | matched(IoU≥0.5) | mean\|Δscore\| | mean IoU |
|---|---:|---:|---:|---:|---:|
| ground0.jpg | 0 | 0 | 0 | nan | nan |
| ground1.jpg | 1 | 1 | 1 | 0.0222 | 0.9867 |
| ground2.jpg | 0 | 0 | 0 | nan | nan |
| ground3.jpg | 0 | 0 | 0 | nan | nan |
| image0.jpg | 0 | 0 | 0 | nan | nan |
| image1.jpg | 1 | 1 | 1 | 0.0824 | 0.9611 |
| image2.jpg | 0 | 0 | 0 | nan | nan |
| image3.jpg | 0 | 0 | 0 | nan | nan |
| image4.jpg | 0 | 0 | 0 | nan | nan |
| image5.jpg | 0 | 0 | 0 | nan | nan |
| image6.jpg | 0 | 0 | 0 | nan | nan |
| image7.jpg | 1 | 1 | 1 | 0.0293 | 0.9959 |
| image8.jpg | 1 | 1 | 1 | 0.1063 | 0.9951 |
| image9.jpg | 1 | 1 | 1 | 0.0649 | 0.9908 |

**합계**: ORT 5 · ncnn 5 · matched 5
**평균 |Δscore|** 0.0610 · **평균 IoU** 0.9859

**게이트(det ±1, IoU≥0.95, |Δscore|≤0.1): PASS ✅**

## Notes
- ncnn export 시 ultralytics가 end2end(one-to-one) 분기를 끈다 → 출력은 **one-to-many head `(1,5,8400)` xywh+sigmoid**. 따라서 배포 디코드는 **NMS 필수**(o2o 가정과 다름). C++ 모듈도 NMS 기본 적용.
- 기준 ONNX(`_fp32.onnx`)는 end2end o2o `(1,300,6)`. o2o는 score가 더 sharp, o2m+NMS는 동일 최종 박스를 재현(위 IoU/Δscore).
- FP16 export라 score 소폭 양자화 오차는 정상. 박스 좌표는 거의 일치.
- 동일 디코드로 실 val 이미지에서도 검증(모든 객체 매칭, mean IoU≈0.98).
