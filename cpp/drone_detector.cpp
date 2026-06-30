#include "drone_detector.h"

#include <algorithm>
#include <chrono>
#include <cmath>

#include "net.h"
#include "mat.h"

namespace dronedet {

namespace {

float iou(const Det& a, const Det& b) {
    const float xa = std::max(a.x1, b.x1), ya = std::max(a.y1, b.y1);
    const float xb = std::min(a.x2, b.x2), yb = std::min(a.y2, b.y2);
    const float inter = std::max(0.f, xb - xa) * std::max(0.f, yb - ya);
    const float aa = std::max(0.f, a.x2 - a.x1) * std::max(0.f, a.y2 - a.y1);
    const float ab = std::max(0.f, b.x2 - b.x1) * std::max(0.f, b.y2 - b.y1);
    return inter / (aa + ab - inter + 1e-9f);
}

// class-agnostic NMS, score 내림차순.
std::vector<Det> nms(std::vector<Det> dets, float thr) {
    std::sort(dets.begin(), dets.end(),
              [](const Det& a, const Det& b) { return a.score > b.score; });
    std::vector<Det> keep;
    std::vector<char> removed(dets.size(), 0);
    for (size_t i = 0; i < dets.size(); ++i) {
        if (removed[i]) continue;
        keep.push_back(dets[i]);
        for (size_t j = i + 1; j < dets.size(); ++j)
            if (!removed[j] && iou(dets[i], dets[j]) >= thr) removed[j] = 1;
    }
    return keep;
}

}  // namespace

DroneDetectorNcnn::DroneDetectorNcnn(bool use_vulkan)
    : net_(new ncnn::Net()), use_vulkan_(use_vulkan) {
    // opt 는 load 전에 설정해야 한다(특히 vulkan allocator).
    net_->opt.use_vulkan_compute = use_vulkan_;
    net_->opt.use_fp16_packed = true;
    net_->opt.use_fp16_storage = true;
    net_->opt.use_fp16_arithmetic = true;
}

DroneDetectorNcnn::~DroneDetectorNcnn() { delete net_; }

bool DroneDetectorNcnn::load(const std::string& param_path,
                             const std::string& bin_path) {
    if (net_->load_param(param_path.c_str()) != 0) return false;
    if (net_->load_model(bin_path.c_str()) != 0) return false;
    return true;
}

std::vector<Det> DroneDetectorNcnn::detect(const cv::Mat& bgr, float conf,
                                           float nms_iou, bool map_to_original) {
    const int img_w = bgr.cols, img_h = bgr.rows;
    const int target = 640;

    // letterbox: Python scripts/parity_ncnn.py 와 동일 계산.
    const float r = std::min(static_cast<float>(target) / img_h,
                             static_cast<float>(target) / img_w);
    const int nw = static_cast<int>(std::round(img_w * r));
    const int nh = static_cast<int>(std::round(img_h * r));
    const int top = (target - nh) / 2;
    const int left = (target - nw) / 2;
    const int bottom = target - nh - top;
    const int right = target - nw - left;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(
        bgr.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, nw, nh);
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, top, bottom, left, right,
                           ncnn::BORDER_CONSTANT, 114.f);
    const float norm[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in_pad.substract_mean_normalize(nullptr, norm);

    // 추론 구간만 계측(전·후처리 제외). Vulkan extract 는 블로킹이라 벽시계로 GPU 포함.
    const auto t0 = std::chrono::high_resolution_clock::now();
    ncnn::Extractor ex = net_->create_extractor();
    ex.input("in0", in_pad);
    ncnn::Mat out;
    ex.extract("out0", out);
    const auto t1 = std::chrono::high_resolution_clock::now();
    last_infer_ms_ =
        std::chrono::duration<double, std::milli>(t1 - t0).count();

    // out: (w=8400, h=5) — row(0..3)=cx,cy,w,h, row(4)=score. (640 입력 좌표)
    const int n = out.w;
    const float* pcx = out.row(0);
    const float* pcy = out.row(1);
    const float* pw = out.row(2);
    const float* ph = out.row(3);
    const float* ps = out.row(4);

    std::vector<Det> cand;
    cand.reserve(64);
    for (int i = 0; i < n; ++i) {
        const float s = ps[i];
        if (s < conf) continue;
        const float cx = pcx[i], cy = pcy[i], w = pw[i], h = ph[i];
        cand.push_back({cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, s, 0});
    }
    std::vector<Det> dets = nms(std::move(cand), nms_iou);

    if (map_to_original) {  // letterbox 역변환 → 원본 좌표
        for (Det& d : dets) {
            d.x1 = std::min(std::max((d.x1 - left) / r, 0.f),
                            static_cast<float>(img_w));
            d.y1 = std::min(std::max((d.y1 - top) / r, 0.f),
                            static_cast<float>(img_h));
            d.x2 = std::min(std::max((d.x2 - left) / r, 0.f),
                            static_cast<float>(img_w));
            d.y2 = std::min(std::max((d.y2 - top) / r, 0.f),
                            static_cast<float>(img_h));
        }
    }
    return dets;
}

}  // namespace dronedet
