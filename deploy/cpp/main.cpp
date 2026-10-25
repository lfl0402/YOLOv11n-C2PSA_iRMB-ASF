// 桑叶病虫害检测演示程序(需要 OpenCV)。
//
// 用法:
//   mulberry_detect <param> <bin> <图片路径> [--out 目录]
//   mulberry_detect <param> <bin> <图片目录> [--out 目录]          # 批量, 目录内 jpg/png 全部检测
//   mulberry_detect <param> <bin> --cam 0 [--out 目录]            # 摄像头实时(Q 键退出)
//   mulberry_detect <param> <bin> --cam 0 --save-video out.mp4    # 摄像头并录制
//
// 通用参数: --conf 0.25 --nms 0.45 --threads 4 --size 640
// 说明: 树莓派 4B 建议 --threads 4; 若 INT8 模型则把 <bin> 换成 ncnn2int8 的产物并加 --int8。

#include <stdio.h>
#include <chrono>
#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio/videoio.hpp>

#include "yolov11.h"

static void draw(cv::Mat& img, const std::vector<DetBox>& boxes) {
    static const cv::Scalar kColors[6] = {
        cv::Scalar(0, 0, 255), cv::Scalar(0, 255, 0), cv::Scalar(255, 0, 0),
        cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 255), cv::Scalar(255, 255, 0),
    };
    for (const DetBox& b : boxes) {
        const cv::Scalar color = kColors[b.label % 6];
        cv::rectangle(img, cv::Rect(cvRound(b.x0), cvRound(b.y0), cvRound(b.x1 - b.x0), cvRound(b.y1 - b.y0)),
                      color, 2);
        char text[128];
        snprintf(text, sizeof(text), "%s %.2f", kClassNames[b.label], b.score);
        cv::putText(img, text, cv::Point(cvRound(b.x0), std::max(12, cvRound(b.y0) - 6)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <param> <bin> <image|dir> [--out dir] [--conf c] [--nms n] [--threads t]\n"
                        "       %s <param> <bin> --cam <id> [--out dir] [--save-video out.mp4]\n",
                argv[0], argv[0]);
        return -1;
    }

    std::string param = argv[1];
    std::string bin = argv[2];
    std::string input = argv[3];
    std::string out_dir;
    std::string video_out;
    int threads = 4;
    float conf = 0.25f;
    float nms = 0.45f;
    bool int8 = false;

    for (int i = 4; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : std::string(); };
        if (a == "--out") out_dir = next();
        else if (a == "--conf") conf = static_cast<float>(atof(next().c_str()));
        else if (a == "--nms") nms = static_cast<float>(atof(next().c_str()));
        else if (a == "--threads") threads = atoi(next().c_str());
        else if (a == "--int8") int8 = true;
        else if (a == "--save-video") video_out = next();
    }

    MulberryDetector det;
    det.conf_thresh = conf;
    det.nms_thresh = nms;
    if (det.load(param, bin, threads, int8) != 0) return -1;

    if (input == "--cam") {
        int cam_id = (argc > 4) ? atoi(argv[4]) : 0;
        cv::VideoCapture cap(cam_id);
        if (!cap.isOpened()) {
            fprintf(stderr, "[error] 无法打开摄像头 %d\n", cam_id);
            return -1;
        }
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        cv::VideoWriter writer;
        if (!video_out.empty()) {
            writer.open(video_out, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 15,
                        cv::Size(cvRound(cap.get(cv::CAP_PROP_FRAME_WIDTH)), cvRound(cap.get(cv::CAP_PROP_FRAME_HEIGHT))));
        }
        cv::Mat frame;
        while (cap.read(frame)) {
            auto t0 = std::chrono::steady_clock::now();
            std::vector<DetBox> boxes = det.detect(frame.data, frame.cols, frame.rows);
            auto t1 = std::chrono::steady_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            draw(frame, boxes);
            char text[64];
            snprintf(text, sizeof(text), "%.1f ms  %.1f FPS  %d obj", ms, 1000.0 / ms, (int)boxes.size());
            cv::putText(frame, text, cv::Point(8, 22), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
            if (writer.isOpened()) writer.write(frame);
            cv::imshow("mulberry", frame);
            if (cv::waitKey(1) == 'q') break;
        }
        return 0;
    }

    // 图片 / 目录模式
    if (!out_dir.empty()) std::filesystem::create_directories(out_dir);  // 否则 imwrite 会静默失败
    std::vector<std::string> files;
    std::vector<cv::String> found;
    if (cv::imread(input).empty()) {
        cv::glob(input + "/*.jpg", found);
        std::vector<cv::String> pngs;
        cv::glob(input + "/*.png", pngs);
        found.insert(found.end(), pngs.begin(), pngs.end());
        for (const auto& f : found) files.push_back(f);
        if (files.empty()) {
            fprintf(stderr, "[error] 目录里没有 jpg/png 图片: %s\n", input.c_str());
            return -1;
        }
    } else {
        files.push_back(input);
    }

    double total_ms = 0;
    for (const std::string& f : files) {
        cv::Mat img = cv::imread(f);
        if (img.empty()) {
            fprintf(stderr, "[warn] 读图失败: %s\n", f.c_str());
            continue;
        }
        auto t0 = std::chrono::steady_clock::now();
        std::vector<DetBox> boxes = det.detect(img.data, img.cols, img.rows);
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_ms += ms;

        printf("%-40s  %.1f ms  检出 %d 个目标\n", f.c_str(), ms, (int)boxes.size());
        for (const DetBox& b : boxes)
            printf("    %-28s conf=%.3f  (%d,%d,%d,%d)\n", kClassNames[b.label], b.score,
                   cvRound(b.x0), cvRound(b.y0), cvRound(b.x1), cvRound(b.y1));

        if (!out_dir.empty()) {
            draw(img, boxes);
            std::string out_name = out_dir + "/" + f.substr(f.find_last_of("/\\") + 1);
            cv::imwrite(out_name, img);
        }
    }
    if (files.size() > 1)
        printf("共 %d 张, 平均单帧 %.1f ms(%.1f FPS, 不含读图/写图 IO)\n",
               (int)files.size(), total_ms / files.size(), files.size() * 1000.0 / total_ms);
    return 0;
}
