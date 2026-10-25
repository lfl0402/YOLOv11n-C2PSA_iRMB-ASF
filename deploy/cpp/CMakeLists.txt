cmake_minimum_required(VERSION 3.13)
project(mulberry_ncnn CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ncnn REQUIRED)

# 树莓派 4B: Cortex-A72 是 ARMv8.0 —— 千万不要加 NCNN_ARM82 相关宏或 -mcpu=cortex-a76 等
# ARMv8.2 指令, 否则运行会直接 SIGILL 崩溃。
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -mcpu=cortex-a72 -fopenmp")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -fopenmp")
endif()

add_library(mulberry_detector STATIC yolov11.cpp)
target_include_directories(mulberry_detector PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mulberry_detector ncnn)

# 纯推理基准: 无第三方依赖
add_executable(mulberry_bench bench.cpp)
target_link_libraries(mulberry_bench mulberry_detector)

# 图片/摄像头检测: 需要 OpenCV(树莓派: sudo apt install libopencv-dev)
find_package(OpenCV QUIET COMPONENTS core imgproc highgui videoio imgcodecs)
if(OpenCV_FOUND)
    add_executable(mulberry_detect main.cpp)
    target_link_libraries(mulberry_detect mulberry_detector ${OpenCV_LIBS})
    message(STATUS "OpenCV 已找到 -> 生成 mulberry_detect(图片/目录/摄像头)")
else()
    message(STATUS "未找到 OpenCV -> 仅生成 mulberry_bench; 需要检测程序请先 sudo apt install libopencv-dev")
endif()
