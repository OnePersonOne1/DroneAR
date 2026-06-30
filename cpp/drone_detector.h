// DroneDetectorNcnn — ncnn-Vulkan 추론 래퍼 (YOLO26n drone, imgsz=640).
//
// 입력 ncnn 모델: weights/yolo26n_drone_640_ncnn_model (FP16 export, one-to-many head).
//   출력 out0 = (5, 8400): [cx, cy, w, h, sigmoid_score] (640 letterbox-입력 픽셀 좌표).
//   end2end(o2o) 분기는 ncnn export 시 비활성 → 디코드 단계에서 NMS 필수.
//
// 전처리는 Python 파이프라인(scripts/parity_ncnn.py letterbox)과 수치 일치:
//   aspect-preserving resize → pad 114 → BGR2RGB → /255.
#pragma once
#include <string>
#include <vector>

#include "simpleocv.h"   // ncnn 번들 cv::Mat/imread (NCNN_SIMPLEOCV=ON)

namespace ncnn { class Net; }

namespace dronedet {

struct Det {
    float x1, y1, x2, y2, score;
    int cls;
};

class DroneDetectorNcnn {
public:
    // use_vulkan=false 면 CPU 폴백(동일 그래프). 생성자에서 opt 설정.
    explicit DroneDetectorNcnn(bool use_vulkan = true);
    ~DroneDetectorNcnn();
    DroneDetectorNcnn(const DroneDetectorNcnn&) = delete;
    DroneDetectorNcnn& operator=(const DroneDetectorNcnn&) = delete;

    // param/bin 경로 로드. 성공 시 true.
    bool load(const std::string& param_path, const std::string& bin_path);

    // BGR 이미지 추론. 반환 박스 좌표:
    //   map_to_original=false → 640 letterbox-입력 좌표(파이썬 parity 기준과 동일 공간)
    //   map_to_original=true  → 원본 이미지 좌표
    std::vector<Det> detect(const cv::Mat& bgr, float conf = 0.25f,
                            float nms_iou = 0.7f, bool map_to_original = false);

    // 직전 detect() 추론 구간(전처리 후 input~extract) 벽시계 ms.
    double last_infer_ms() const { return last_infer_ms_; }
    bool gpu_enabled() const { return use_vulkan_; }

private:
    ncnn::Net* net_;
    bool use_vulkan_;
    double last_infer_ms_ = 0.0;
};

}  // namespace dronedet
