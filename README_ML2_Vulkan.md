# ML2 GPU(Vulkan) 추론 경로 — ncnn-Vulkan

ML2 현재 추론은 ONNX Runtime **CPU**(XNNPACK) 경로이며 약 **7 FPS**다. 이 문서는 RDNA2 iGPU 를
**ncnn-Vulkan** 으로 활용하는 GPU 추론 경로를 정리한다. 대상 모델은 `yolo26n_drone_640`,
정밀도 FP16.

> **검증 범위 주의.** ML2 기기가 없어 on-device 측정은 하지 않았다. 추론 코드 정확성과 Vulkan
> backend 동작·latency 는 **호스트 RTX 4090(Vulkan)** 으로 검증했다(RDNA2 와 동일 ncnn Vulkan
> backend). ML2의 RDNA2 실측 성능만 on-device 남은 과제 항목으로
>남는다 → [docs/ML2_ONDEVICE_RUNBOOK.md](docs/ML2_ONDEVICE_RUNBOOK.md).

## 왜 ncnn-Vulkan 인가

- ML2 = AMD RDNA2 → **NVIDIA 아님**. TensorRT/CUDA, DirectML(Windows), ROCm(데이터센터/Linux) 모두 부적합.
- ORT NNAPI EP 는 AMD GPU용 NNAPI HAL 없으면 CPU 폴백.
- **ncnn-Vulkan = 벤더 무관 Vulkan compute** → AMD 포함 동작. RDNA2 에 가장 현실적인 GPU 경로.

## 변환 — `.pt → ncnn`

```bash
.venv/bin/python -c "from ultralytics import YOLO; \
  YOLO('weights/yolo26n_drone_640.pt').export(format='ncnn', half=True, imgsz=640, batch=1)"
# -> weights/yolo26n_drone_640_ncnn_model/{model.ncnn.param, model.ncnn.bin}  (FP16)
```

- 출력 `out0 = (5, 8400)` = `[cx, cy, w, h, sigmoid_score]`, 640 letterbox-입력 좌표.
- ⚠️ ncnn export 시 ultralytics 가 **end2end(one-to-one) 분기를 끈다** → **one-to-many head**.
  따라서 디코드에 **NMS 필수**(NMS-free 가정과 다름). C++ 모듈이 IoU 0.7 class-agnostic NMS 적용.

## Parity — ncnn vs ONNX Runtime(FP32)

`scripts/parity_ncnn.py` 가 동일 letterbox 전처리로 ncnn(CPU) 박스와 ORT `_fp32.onnx`(o2o
`(1,300,6)`) 기준 박스를 비교한다(640 좌표).

```bash
.venv/bin/python scripts/parity_ncnn.py     # -> weights/parity_ncnn.md, parity_ref.{json,csv}
```

게이트(det ±1, mean IoU ≥ 0.95, mean|Δscore| ≤ 0.1) **PASS**. demo 10장 + 실 val 이미지에서
모든 객체 매칭, mean IoU ≈ 0.98. (FP16 export 라 score 소폭 오차는 정상.)

## C++ 추론 모듈 — `cpp/`

- `cpp/drone_detector.{h,cpp}` — `DroneDetectorNcnn`. Vulkan(fp16) 토글, letterbox(파이썬 일치),
  one-to-many 디코드 + NMS, `detect()`/`last_infer_ms()`.
- `cpp/test_host.cpp` — self-test: ORT 기준(`parity_ref.csv`) 대조 + 4090 Vulkan latency.

### 호스트(4090 Vulkan) 빌드·검증 — 검증됨 ✅

```bash
# ncnn (Vulkan, simpleocv) host 빌드 후:
cd cpp && mkdir build && cd build
cmake -Dncnn_DIR=<ncnn>/build-host/install/lib/cmake/ncnn -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j && ./dronedet_selftest ../../demo
```

결과(호스트 **RTX 4090 Vulkan**, ML2 아님):
- Vulkan device + `use_vulkan=1` 동작.
- parity `RESULT: PASS` (matched 5/5, meanIoU ≈ 0.982, mean|Δscore| ≈ 0.025).
- forward latency ≈ **4.4 ± 0.9 ms (≈ 226 FPS)**, imgsz=640 batch=1 warmup=30 iters=200.
- ⚠️ 위 수치는 **호스트 4090** 검증용이며 **ML2 RDNA2 수치가 아니다.**

### ML2(NDK x86_64, Android 10) 빌드 — 빌드 검증됨, on-device 미검증

```bash
export ANDROID_NDK=<ndk-r26+>
cd cpp && mkdir build-ml2 && cd build-ml2
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
      -DANDROID_ABI=x86_64 -DANDROID_PLATFORM=android-29 \
      -Dncnn_DIR=<ncnn>/build-ml2/install/lib/cmake/ncnn -DDRONEDET_SHARED=ON ..
cmake --build . -j            # -> libdronedet.so (ELF x86-64), dronedet_selftest (PIE)
```

- ncnn ML2 빌드는 **AVX-512 OFF**(ML2 Zen2 미지원, NDK clang ICE 회피) + Vulkan ON.
- on-device 실행/측정은 [runbook](docs/ML2_ONDEVICE_RUNBOOK.md) 참조.

## 앱 연동

MLSDK 카메라/렌더 글루 계약은 [cpp/mlsdk_glue.md](cpp/mlsdk_glue.md). detect-then-track 보간은 후속 작업.

## 폴백 순위

1. ML2 에서 Vulkan compute queue 미노출 → **MNN(Vulkan/OpenCL)** 시도(OpenCL ICD 있으면).
2. 그래도 불가 → **CPU 경로 유지**(현 ORT/XNNPACK) + detect-then-track·파이프라인 오버랩.

## 검증됨 vs 미검증 요약

| 항목 | 상태 |
|---|---|
| `.pt → ncnn` FP16 변환 | ✅ |
| ncnn vs ORT parity (CPU, demo+val) | ✅ PASS |
| C++ Vulkan 추론 정확성 (4090, parity) | ✅ PASS |
| C++ Vulkan latency (4090) | ✅ 측정 (≈226 FPS, **ML2 아님**) |
| ML2 NDK x86_64 빌드(.so/selftest, ELF 확인) | ✅ |
| ML2 RDNA2 Vulkan 노출·실측 FPS | ⬜ on-device 필요(runbook) |
