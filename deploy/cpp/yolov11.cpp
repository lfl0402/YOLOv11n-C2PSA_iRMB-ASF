// YOLOv11n-C2PSA_iRMB-ASF(桑叶病虫害) NCNN 推理实现。
// 前处理/后处理与 deploy/export.py 严格一致, 保证树莓派端与 PyTorch 端检测结果可复现。

#include "yolov11.h"

#include <math.h>
#include <stdio.h>
#include <algorithm>
#include <cstring>
#include <vector>

namespace {

// 输入像素为 BGR(cv::imread 输出)。模型要求 **RGB 通道 + 0~1 范围**(与 ultralytics
// 预测端一致, 已用逐项比对确认), 因此这里同时完成通道重排与归一化。
// 不使用 ncnn::Mat::from_pixels, 避免其不归一化的默认行为。
ncnn::Mat bgr_to_mat(const unsigned char* bgr, int w, int h) {
    ncnn::Mat m(w, h, 3);
    const float inv = 1.0f / 255.0f;
    for (int y = 0; y < h; ++y) {
        const unsigned char* row = bgr + static_cast<size_t>(y) * w * 3;
        // 新版 ncnn 的 Mat::channel() 返回 Mat 视图, 需用 .row(y) 取行指针
        float* r = m.channel(0).row(y);  // R <- 源 BGR 的第 3 字节
        float* g = m.channel(1).row(y);  // G <- 第 2 字节
        float* b = m.channel(2).row(y);  // B <- 第 1 字节
        for (int x = 0; x < w; ++x) {
            r[x] = row[x * 3 + 2] * inv;
            g[x] = row[x * 3 + 1] * inv;
            b[x] = row[x * 3 + 0] * inv;
        }
    }
    return m;
}

// 等比缩放 + 114 灰边填充, 与 export.py 的 letterbox() 一致。
// 返回填充后的 640x640 图像、缩放比、左上角填充量。
void letterbox(const unsigned char* bgr, int img_w, int img_h, int size,
               unsigned char* padded, float* scale, int* pad_l, int* pad_t) {
    const float s = static_cast<float>(size) / std::max(img_w, img_h);
    const int nw = static_cast<int>(std::round(img_w * s));
    const int nh = static_cast<int>(std::round(img_h * s));
    const int pl = (size - nw) / 2;
    const int pt = (size - nh) / 2;

    std::memset(padded, 114, size * size * 3);
    std::vector<unsigned char> resized(static_cast<size_t>(nw) * nh * 3);
    ncnn::resize_bilinear_c3(bgr, img_w, img_h, resized.data(), nw, nh);
    for (int y = 0; y < nh; ++y)
        std::memcpy(padded + ((pt + y) * size + pl) * 3, resized.data() + y * nw * 3, nw * 3);

    *scale = s;
    *pad_l = pl;
    *pad_t = pt;
}

float box_iou(const DetBox& a, const DetBox& b) {
    const float ix0 = std::max(a.x0, b.x0), iy0 = std::max(a.y0, b.y0);
    const float ix1 = std::min(a.x1, b.x1), iy1 = std::min(a.y1, b.y1);
    const float inter = std::max(0.f, ix1 - ix0) * std::max(0.f, iy1 - iy0);
    const float area_a = (a.x1 - a.x0) * (a.y1 - a.y0);
    const float area_b = (b.x1 - b.x0) * (b.y1 - b.y0);
    return inter / std::max(1e-9f, area_a + area_b - inter);
}

}  // namespace

int MulberryDetector::load(const std::string& param_path, const std::string& bin_path,
                           int num_threads, bool use_int8) {
    net_.opt.num_threads = num_threads;
    net_.opt.lightmode = true;
    net_.opt.use_vulkan_compute = false;
    // 树莓派 4B(Cortex-A72, ARMv8.0)没有 FP16 算术与 INT8 点积指令:
    net_.opt.use_fp16_packed = false;
    net_.opt.use_fp16_storage = false;
    net_.opt.use_fp16_arithmetic = false;
    net_.opt.use_int8_inference = use_int8;

    if (net_.load_param(param_path.c_str()) != 0) {
        fprintf(stderr, "[error] load_param 失败: %s\n", param_path.c_str());
        return -1;
    }
    if (net_.load_model(bin_path.c_str()) != 0) {
        fprintf(stderr, "[error] load_model 失败: %s\n", bin_path.c_str());
        return -1;
    }
    return 0;
}

void MulberryDetector::infer(const ncnn::Mat& in, ncnn::Mat& out) {
    ncnn::Extractor ex = net_.create_extractor();
    ex.input("in0", in);
    // pnnx 对单一输出通常命名为 out0; 少数版本命名不同时用 Netron 查看并修改这里。
    if (ex.extract("out0", out) != 0)
        fprintf(stderr, "[error] extract(\"out0\") 失败, 请用 Netron 查看模型输出名\n");
}

std::vector<DetBox> MulberryDetector::detect(const unsigned char* bgr, int img_w, int img_h) {
    const int size = input_size;
    std::vector<unsigned char> padded(static_cast<size_t>(size) * size * 3);
    float scale = 1.f;
    int pad_l = 0, pad_t = 0;
    letterbox(bgr, img_w, img_h, size, padded.data(), &scale, &pad_l, &pad_t);

    ncnn::Mat in = bgr_to_mat(padded.data(), size, size);
    ncnn::Mat out;
    infer(in, out);

    // out: (1, 4+nc, 8400), w=8400, h=10, c=1。行 0..3 = cx,cy,w,h; 行 4..9 = 类别概率。
    const float* ptr = out.channel(0);
    const int w = out.w;

    std::vector<DetBox> cands;
    cands.reserve(128);
    for (int j = 0; j < w; ++j) {
        int best = -1;
        float best_score = conf_thresh;
        for (int k = 0; k < nc_; ++k) {
            const float s = ptr[(4 + k) * w + j];
            if (s > best_score) {
                best_score = s;
                best = k;
            }
        }
        if (best < 0) continue;

        const float cx = ptr[j];
        const float cy = ptr[w + j];
        const float bw = ptr[2 * w + j];
        const float bh = ptr[3 * w + j];

        // (cx, cy, w, h) -> xyxy -> 原图坐标(去掉 letterbox)
        float x0 = (cx - bw * 0.5f - pad_l) / scale;
        float y0 = (cy - bh * 0.5f - pad_t) / scale;
        float x1 = (cx + bw * 0.5f - pad_l) / scale;
        float y1 = (cy + bh * 0.5f - pad_t) / scale;
        x0 = std::max(0.f, std::min(x0, static_cast<float>(img_w)));
        y0 = std::max(0.f, std::min(y0, static_cast<float>(img_h)));
        x1 = std::max(0.f, std::min(x1, static_cast<float>(img_w)));
        y1 = std::max(0.f, std::min(y1, static_cast<float>(img_h)));

        cands.push_back(DetBox{best, best_score, x0, y0, x1, y1});
    }

    // 分数降序 + 贪心 NMS, 与 export.py 的 decode_out() 一致
    std::sort(cands.begin(), cands.end(),
              [](const DetBox& a, const DetBox& b) { return a.score > b.score; });
    std::vector<DetBox> keep;
    for (const DetBox& c : cands) {
        bool suppressed = false;
        for (const DetBox& k : keep) {
            if (box_iou(c, k) > nms_thresh) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) keep.push_back(c);
    }
    return keep;
}
