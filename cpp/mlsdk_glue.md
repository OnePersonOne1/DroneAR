# MLSDK 글루 — 인터페이스 계약 (스텁)

실제 ML2 앱은 Magic Leap MLSDK C API 로 카메라 취득·AR 렌더를 한다. 그 부분은 **이 리포 범위 밖**
(ML2 팀/앱 코드). 여기서는 `dronedet::DroneDetectorNcnn` 과 앱 사이 계약만 정의한다.

## 입력: ML camera frame → 추론 입력

- ML camera 콜백은 보통 **YUV(NV12 등)** 프레임 + width/height/stride 를 준다.
- 변환 지점: YUV → BGR(또는 RGB) 연속 버퍼 → `cv::Mat`(simpleocv) 또는 직접
  `ncnn::Mat::from_pixels`. `DroneDetectorNcnn::detect()` 는 **BGR `cv::Mat`** 을 받는다.
- letterbox/정규화는 `detect()` 내부에서 처리(전처리 일치 보장). 앱은 색공간 변환만 책임.

```cpp
// 의사코드
cv::Mat bgr = yuv_to_bgr(frame.data, frame.w, frame.h, frame.stride);
auto dets = detector.detect(bgr, /*conf=*/0.25f, /*nms_iou=*/0.7f,
                            /*map_to_original=*/true);  // 원본 프레임 좌표
```

## 출력: Det → AR overlay

- `Det{ x1,y1,x2,y2,score,cls }` (cls=0 drone). `map_to_original=true` 면 카메라 프레임
  픽셀 좌표.
- 앱은 카메라 intrinsics/extrinsics 로 프레임 좌표 → 월드/스테레오 디스플레이 좌표 매핑 후
  박스 + 라벨(score, 등록 정보) 렌더.

## 권장 파이프라인 — detect-then-track

- detection 은 ncnn-Vulkan 으로 N FPS(측정값), overlay 는 디스플레이 refresh(예: 120Hz)로.
- 사이 프레임은 **추적으로 보간**(Kalman + IoU/optical-flow). 추론 주기와 렌더 주기를 분리해
  GPU/CPU 예산과 AR 부드러움을 양립.
- 추적기/보간은 **별도 후속 작업**(이 리포 미포함).

## 수명·스레딩

- `DroneDetectorNcnn` 는 1 인스턴스를 추론 스레드 1개가 소유(extractor 는 호출마다 생성).
- ncnn 메모리 수명 주의: 입력 버퍼는 `detect()` 반환까지 유효해야 함(내부에서 복사 처리).
