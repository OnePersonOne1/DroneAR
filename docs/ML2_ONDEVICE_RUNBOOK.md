# ML2 on-device 검증 runbook — ncnn-Vulkan (RDNA2 iGPU)

ML2 기기 보유자가 따라 실행하는 체크리스트.
ML2 를 보유하지 않아서, 공개된 스펙을 바탕으로 작성하였다. 이에 부정확한 부분이 있을 수 있음을 미리 밝힌다.

수치 칸은 비워두었다. — 측정 후 채워야 한다.

전제: `cpp/build-ml2` 에서 NDK(x86_64, API29, Vulkan)로 교차컴파일된
`libdronedet.so` + `dronedet_selftest` 와 ncnn 모델
`weights/yolo26n_drone_640_ncnn_model/` (param/bin), `weights/parity_ref.csv` 를 사용한다.

호스트(RTX 4090 Vulkan)에서 추론 정확성 + Vulkan 경로는 이미 검증됨.
ML2에서 실증할 것은 두 가지: **(a) AOSP 빌드가 앱에 Vulkan compute queue 를 노출하는가, (b) RDNA2 실측 성능.**

---

## 0. 전제 — 무엇이 게이트인가

ncnn-Vulkan 은 RDNA2 를 포함한 벤더 무관 Vulkan compute 로 동작한다(HW 지원은 확실).
유일한 리스크는 **ML2 AOSP 가 사용자 앱에 Vulkan compute queue family 를 노출하는지**다.
노출되면 GPU 추론 성립, 막혀 있으면 폴백(§5)으로 전환.

## 1. Vulkan compute 노출 확인

```bash
adb push libdronedet.so dronedet_selftest /data/local/tmp/dronedet/
adb push weights/yolo26n_drone_640_ncnn_model /data/local/tmp/dronedet/model
adb push weights/parity_ref.csv demo /data/local/tmp/dronedet/
adb shell "cd /data/local/tmp/dronedet && LD_LIBRARY_PATH=. ./dronedet_selftest ./demo ./model ./parity_ref.csv"
```

- selftest 시작 로그에 `[0 <AMD RDNA2 ...>]` 형태로 **physical device + compute queue** 가
  열거되면 노출 OK. CPU(llvmpipe)만 보이거나 device 0개면 노출 막힘 → §5.
- 보조 확인: `adb shell vulkaninfo` 가 있으면 `VkPhysicalDevice` 와
  `queueFlags: ... COMPUTE` 존재 확인.

| 점검 | 결과(채울 것) |
|---|---|
| Vulkan physical device 열거 | ☐ AMD ☐ CPU만 ☐ 없음 |
| compute queue family 존재 | ☐ |
| selftest `use_vulkan=1` 동작 | ☐ |

## 2. OpenCL ICD 존재 여부(대안 경로)

```bash
adb shell "ls /vendor/lib64/libOpenCL.so /system/lib64/libOpenCL.so 2>/dev/null"
```
존재하면 Vulkan 이 막혔을 때 MNN-OpenCL 대안(§5)이 열린다. (결과: ☐ 있음 ☐ 없음)

## 3. Parity 확인

`dronedet_selftest` 의 parity 섹션이 동봉 ref(`parity_ref.csv`, ORT FP32 기준)와
`RESULT: PASS` 인지 확인. 호스트와 동일 기준이므로 ML2 에서도 PASS 여야 한다.

- parity 결과: ☐ PASS ☐ FAIL (meanIoU ____ , mean|Δscore| ____)

## 4. ML2 GPU latency·FPS 측정 (현재 CPU 경로와 직접 비교)

`dronedet_selftest` 의 latency 섹션(warmup=30, iters=200, imgsz=640, batch=1)
출력값을 기입한다.

| 항목 | 값(채울 것) |
|---|---|
| RDNA2 Vulkan forward mean±std (ms) | ____ ± ____ |
| RDNA2 Vulkan FPS | ____ |
| (참고) 현재 ONNX Runtime **CPU** 경로 FPS | 약 7 |

> 비교 표의 CPU 7 FPS 는 ML2 CPU 실측값이다. GPU 값은 측정 후 기입.

## 5. Vulkan 미노출 시 폴백 순위

1. **MNN (Vulkan/OpenCL backend)** — §2 에서 OpenCL ICD 가 있으면 우선 시도.
   ncnn 모델 대신 MNN 변환 필요. RDNA2 OpenCL 로 GPU 추론.
2. 그래도 불가 → **CPU 경로 유지**(현 ONNX Runtime/XNNPACK) + 다른 축 최적화:
   - detect-then-track(§ `cpp/mlsdk_glue.md`): 매 프레임 추론 대신 추론+추적 보간.
   - 파이프라인 오버랩(추론과 렌더 비동기), 입력 해상도/모델 크기 조정.

## 부록 — AR 자원 경합 주의

RDNA2 iGPU 는 120Hz AR 스테레오 렌더링과 자원을 공유한다. GPU 추론 도입 시
렌더 프레임 예산과의 경합을 프로파일링해야 한다(추론을 GPU 로 올릴지 여부의 핵심 트레이드오프).