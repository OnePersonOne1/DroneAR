// 호스트(RTX 4090 Vulkan) self-test:
//  (1) ncnn-Vulkan 추론이 Python parity 기준(weights/parity_ref.csv, ORT FP32)과 일치하는지 검증
//  (2) 4090 Vulkan forward latency 측정
//
// 중요: 출력 latency 는 *호스트 RTX 4090* 수치이며 *ML2 수치가 아니다*.
//       ML2(RDNA2)는 별도 빌드(cpp/build-ml2)로 기기에서만 측정한다(docs/ML2_ONDEVICE_RUNBOOK.md).
//
// 사용: ./test_host [demo_dir] [ncnn_model_dir] [parity_ref.csv]
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "drone_detector.h"

using dronedet::Det;
using dronedet::DroneDetectorNcnn;

static float iou(const Det& a, const Det& b) {
    const float xa = std::max(a.x1, b.x1), ya = std::max(a.y1, b.y1);
    const float xb = std::min(a.x2, b.x2), yb = std::min(a.y2, b.y2);
    const float inter = std::max(0.f, xb - xa) * std::max(0.f, yb - ya);
    const float aa = std::max(0.f, a.x2 - a.x1) * std::max(0.f, a.y2 - a.y1);
    const float ab = std::max(0.f, b.x2 - b.x1) * std::max(0.f, b.y2 - b.y1);
    return inter / (aa + ab - inter + 1e-9f);
}

// parity_ref.csv -> filename별 기준 박스(640 letterbox 좌표).
static std::map<std::string, std::vector<Det>> load_ref(const std::string& path) {
    std::map<std::string, std::vector<Det>> ref;
    std::ifstream f(path);
    std::string line;
    std::getline(f, line);  // header
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string name, tok;
        std::getline(ss, name, ',');
        Det d{};
        d.cls = 0;
        std::getline(ss, tok, ','); d.x1 = std::stof(tok);
        std::getline(ss, tok, ','); d.y1 = std::stof(tok);
        std::getline(ss, tok, ','); d.x2 = std::stof(tok);
        std::getline(ss, tok, ','); d.y2 = std::stof(tok);
        std::getline(ss, tok, ','); d.score = std::stof(tok);
        ref[name].push_back(d);
    }
    return ref;
}

int main(int argc, char** argv) {
    const std::string demo = argc > 1 ? argv[1] : "../../demo";
    const std::string mdir = argc > 2 ? argv[2]
                                      : "../../weights/yolo26n_drone_640_ncnn_model";
    const std::string refp = argc > 3 ? argv[3]
                                      : "../../weights/parity_ref.csv";
    const float conf = 0.25f;

    DroneDetectorNcnn det(/*use_vulkan=*/true);
    if (!det.load(mdir + "/model.ncnn.param", mdir + "/model.ncnn.bin")) {
        std::fprintf(stderr, "FAILED to load ncnn model from %s\n", mdir.c_str());
        return 2;
    }
    std::printf("ncnn loaded (use_vulkan=%d)\n", det.gpu_enabled());

    auto ref = load_ref(refp);

    // ---- (1) parity ----
    std::printf("\n== parity vs ORT FP32 ref (640 letterbox space) ==\n");
    int n_ref = 0, n_test = 0, n_match = 0, n_cnt_ok = 0, n_imgs = 0;
    double sum_iou = 0, sum_ds = 0;
    for (int i = 0; i < 10; ++i) {
        const std::string name = "image" + std::to_string(i) + ".jpg";
        cv::Mat img = cv::imread(demo + "/" + name);
        if (img.empty()) continue;
        ++n_imgs;
        std::vector<Det> got = det.detect(img, conf, 0.7f, /*map_to_original=*/false);
        const std::vector<Det>& rf = ref[name];
        n_ref += static_cast<int>(rf.size());
        n_test += static_cast<int>(got.size());
        const bool cnt_ok =
            std::abs(static_cast<int>(rf.size()) - static_cast<int>(got.size())) <= 1;
        n_cnt_ok += cnt_ok;
        int m = 0;
        double iiou = 0, ids = 0;
        for (const Det& a : rf) {
            if (got.empty()) continue;
            int bj = 0;
            float bi = -1;
            for (size_t j = 0; j < got.size(); ++j) {
                const float v = iou(a, got[j]);
                if (v > bi) { bi = v; bj = static_cast<int>(j); }
            }
            if (bi >= 0.5f) {
                ++m;
                iiou += bi;
                ids += std::abs(a.score - got[bj].score);
            }
        }
        n_match += m;
        if (m) { sum_iou += iiou; sum_ds += ids; }
        std::printf("  %-12s ref=%zu ncnn=%zu matched=%d meanIoU=%.4f mean|ds|=%.4f %s\n",
                    name.c_str(), rf.size(), got.size(), m,
                    m ? iiou / m : 0.0, m ? ids / m : 0.0,
                    cnt_ok ? "" : "[COUNT MISMATCH]");
    }
    const double mIoU = n_match ? sum_iou / n_match : 0.0;
    const double mDs = n_match ? sum_ds / n_match : 1.0;
    const bool parity_ok =
        (n_cnt_ok == n_imgs) && (mIoU >= 0.95) && (mDs <= 0.10) && (n_match > 0);
    std::printf("  ---- ref=%d ncnn=%d matched=%d | meanIoU=%.4f mean|ds|=%.4f -> %s\n",
                n_ref, n_test, n_match, mIoU, mDs, parity_ok ? "PASS" : "FAIL");

    // ---- (2) latency (RTX 4090 Vulkan; NOT ML2) ----
    std::printf("\n== latency: RTX 4090 Vulkan (host) — NOT ML2 ==\n");
    cv::Mat bench_img = cv::imread(demo + "/image5.jpg");
    if (bench_img.empty()) bench_img = cv::imread(demo + "/image0.jpg");
    const int warmup = 30, iters = 200;
    for (int i = 0; i < warmup; ++i) det.detect(bench_img, conf);
    std::vector<double> ts;
    ts.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        det.detect(bench_img, conf);
        ts.push_back(det.last_infer_ms());
    }
    double mean = 0;
    for (double t : ts) mean += t;
    mean /= ts.size();
    double var = 0;
    for (double t : ts) var += (t - mean) * (t - mean);
    const double sd = std::sqrt(var / ts.size());
    std::printf("  imgsz=640 batch=1 warmup=%d iters=%d\n", warmup, iters);
    std::printf("  forward(host RTX 4090 Vulkan): %.3f +/- %.3f ms  (%.1f FPS)\n",
                mean, sd, 1000.0 / mean);
    std::printf("  [NOTE] 위 수치는 호스트 4090 검증용. ML2 RDNA2 수치 아님.\n");

    std::printf("\nRESULT: %s\n", parity_ok ? "PASS" : "FAIL");
    return parity_ok ? 0 : 1;
}
