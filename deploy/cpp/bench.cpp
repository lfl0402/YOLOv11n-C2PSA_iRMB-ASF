// 纯推理速度基准(不依赖 OpenCV), 用于论文里的 FPS 数据。
// 用法: mulberry_bench <param> <bin> [--threads 4] [--warmup 10] [--runs 50] [--size 640] [--int8]

#include <stdio.h>
#include <chrono>
#include <cstdlib>
#include <string>

#include "net.h"
#include "yolov11.h"

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <param> <bin> [--threads t] [--warmup n] [--runs n] [--size s] [--int8]\n", argv[0]);
        return -1;
    }
    std::string param = argv[1];
    std::string bin = argv[2];
    int threads = 4, warmup = 10, runs = 50, size = 640;
    bool int8 = false;
    for (int i = 3; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : std::string(); };
        if (a == "--threads") threads = atoi(next().c_str());
        else if (a == "--warmup") warmup = atoi(next().c_str());
        else if (a == "--runs") runs = atoi(next().c_str());
        else if (a == "--size") size = atoi(next().c_str());
        else if (a == "--int8") int8 = true;
    }

    MulberryDetector det;
    det.input_size = size;
    if (det.load(param, bin, threads, int8) != 0) return -1;

    ncnn::Mat in(size, size, 3);
    in.fill(0.5f);  // 内容不影响耗时
    ncnn::Mat out;

    for (int i = 0; i < warmup; ++i) det.infer(in, out);

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < runs; ++i) det.infer(in, out);
    auto t1 = std::chrono::steady_clock::now();

    const double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double ms = total_ms / runs;
    printf("输入 %dx%d, 线程 %d, %s\n", size, size, threads, int8 ? "INT8" : "FP32");
    printf("warmup %d 次, 计时 %d 次\n", warmup, runs);
    printf("单帧推理: %.2f ms   (%.1f FPS)\n", ms, 1000.0 / ms);
    return 0;
}
