#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PC 端参考检测: 用 ncnn 模型对给定图片做完整前处理+推理+解码+NMS, 输出检测结果。

用途: 作为树莓派端 mulberry_detect 的比对基准 —— 两边在同一张图上应给出相同的
类别、置信度与框坐标(容差 0.1 像素/0.01 置信度)。

用法:
    python reference.py                     # 默认跑验证集前 3 张
    python reference.py 图片1.jpg 图片2.jpg --out out/
"""
import argparse
import os
import sys

import cv2
import numpy as np
import ncnn

HERE = os.path.dirname(os.path.abspath(__file__))
PARAM = os.path.join(HERE, "model", "yolov11_mulberry.torchscript.ncnn.param")
BIN = os.path.join(HERE, "model", "yolov11_mulberry.torchscript.ncnn.bin")
NAMES = ['Anthracnose', 'Healthy', 'Inchworm', 'Bacterial Wilt',
         'Yellow Melyrid Leaf Beetle', 'Willow Blue Leaf Beetle']
SIZE = 640


def preprocess(img_bgr):
    """等比缩放 + 114 灰边 + BGR→RGB(模型要求 RGB + 0~1)。返回 RGB uint8 与 letterbox 参数。"""
    h, w = img_bgr.shape[:2]
    r = SIZE / max(h, w)
    nh, nw = int(round(h * r)), int(round(w * r))
    canvas = np.full((SIZE, SIZE, 3), 114, dtype=np.uint8)
    pl, pt = (SIZE - nw) // 2, (SIZE - nh) // 2
    canvas[pt:pt + nh, pl:pl + nw] = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(canvas[:, :, ::-1]), r, pl, pt


def detect(net, img_bgr, conf_th=0.25, nms_th=0.45):
    rgb, r, pl, pt = preprocess(img_bgr)
    mat = ncnn.Mat.from_pixels(rgb, ncnn.Mat.PixelType.PIXEL_RGB, SIZE, SIZE)
    mat.substract_mean_normalize([0.0, 0.0, 0.0], [1 / 255.0, 1 / 255.0, 1 / 255.0])
    ex = net.create_extractor()
    ex.input("in0", mat)
    ret, out = ex.extract("out0")
    if ret != 0:
        raise RuntimeError(f"ncnn 推理失败 ret={ret}")
    y = np.array(out).reshape(out.h, out.w)

    cands = []
    for j in range(y.shape[1]):
        s = y[4:, j]
        k = int(s.argmax())
        if s[k] < conf_th:
            continue
        cx, cy, bw, bh = y[:4, j]
        cands.append([k, float(s[k]), (cx - bw / 2 - pl) / r, (cy - bh / 2 - pt) / r,
                      (cx + bw / 2 - pl) / r, (cy + bh / 2 - pt) / r])
    cands.sort(key=lambda c: -c[1])

    def iou(p, q):
        ix0, iy0 = max(p[2], q[2]), max(p[3], q[3])
        ix1, iy1 = min(p[4], q[4]), min(p[5], q[5])
        inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
        return inter / max(1e-9, (p[4] - p[2]) * (p[5] - p[3]) + (q[4] - q[2]) * (q[5] - q[3]) - inter)

    keep = []
    for c in cands:
        if all(iou(c, k) <= nms_th for k in keep):
            keep.append(c)
    return keep


def load_net():
    net = ncnn.Net()
    net.opt.num_threads = 4
    net.opt.use_fp16_packed = False
    net.opt.use_fp16_storage = False
    net.opt.use_fp16_arithmetic = False
    assert net.load_param(PARAM) == 0 and net.load_model(BIN) == 0, "模型加载失败"
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="*", default=[])
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    imgs = a.images or [os.path.join(HERE, "images", f"{n}.jpg") for n in ("47", "48", "49")]
    if a.out:
        os.makedirs(a.out, exist_ok=True)

    net = load_net()
    for p in imgs:
        img = cv2.imread(p)
        if img is None:
            print(f"读图失败: {p}")
            continue
        boxes = detect(net, img)
        print(f"{os.path.basename(p)}  ({img.shape[1]}x{img.shape[0]})  检出 {len(boxes)} 个目标")
        for k, s, x0, y0, x1, y1 in boxes:
            print(f"    {NAMES[k]:28s} conf={s:.3f}  ({x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f})")
        if a.out:
            vis = img.copy()
            for k, s, x0, y0, x1, y1 in boxes:
                cv2.rectangle(vis, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 2)
                cv2.putText(vis, f"{NAMES[k]} {s:.2f}", (int(x0), max(12, int(y0) - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imwrite(os.path.join(a.out, os.path.basename(p)), vis)
    return 0


if __name__ == "__main__":
    sys.exit(main())
