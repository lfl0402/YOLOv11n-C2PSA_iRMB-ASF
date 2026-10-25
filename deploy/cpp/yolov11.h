#ifndef MULBERRY_DETECTOR_H
#define MULBERRY_DETECTOR_H

#include <string>
#include <vector>

#include "net.h"

// 桑叶病虫害 6 类, 顺序与 dataset0224/data.yaml 一致
static const char* kClassNames[6] = {
    "Anthracnose",                 // 0 炭疽病
    "Healthy",                     // 1 健康叶片
    "Inchworm",                    // 2 尺蠖
    "Bacterial Wilt",              // 3 细菌性枯萎
    "Yellow Melyrid Leaf Beetle",  // 4 黄斑叶甲
    "Willow Blue Leaf Beetle",     // 5 柳蓝叶甲
};

struct DetBox {
    int label;
    float score;
    float x0, y0, x1, y1;  // 原图像素坐标
};

// YOLOv11n-C2PSA_iRMB-ASF 的 NCNN 推理封装。
// 模型输出为单一 blob out0: (1, 4+6, 8400), 前 4 行为已解码的 (cx, cy, w, h)(640 空间像素),
// 后 6 行为 sigmoid 类别概率。后处理与 deploy/export.py 的 decode_out() 一一对应。
class MulberryDetector {
public:
    // 加载 .param/.bin。use_int8 仅当 bin 已由 ncnn2int8 量化后置 true。
    int load(const std::string& param_path, const std::string& bin_path,
             int num_threads = 4, bool use_int8 = false);

    // 对 BGR 图像(HWC 交织存放, 与 cv::imread 一致)做检测, 返回原图坐标的框。
    std::vector<DetBox> detect(const unsigned char* bgr, int img_w, int img_h);

    // 直接对 640x640x3 的输入 Mat 推理, 结果写入 out(供 bench / 数值比对)。
    void infer(const ncnn::Mat& in, ncnn::Mat& out);

    int input_size = 640;
    float conf_thresh = 0.25f;
    float nms_thresh = 0.45f;

private:
    ncnn::Net net_;
    int nc_ = 6;
};

#endif  // MULBERRY_DETECTOR_H
