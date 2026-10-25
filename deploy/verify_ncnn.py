# -*- coding: utf-8 -*-
"""端到端数值验证: PyTorch(torchscript) vs ncnn(转换产物)。

注意: torch(Intel OpenMP)与 ncnn 的 OpenMP 在同一进程内共存会段错误,
因此本脚本按模式分进程执行, 结果以 .npy 落盘后再比对:

    python verify_ncnn.py --mode torch    # 独立进程: 跑 torchscript, 输出到 _verify_out/torch/
    python verify_ncnn.py --mode ncnn     # 独立进程: 跑 ncnn,       输出到 _verify_out/ncnn/
    python verify_ncnn.py --mode compare  # 比对两者, 打印最大偏差

若两者输出一致, 说明 导出 → 重写(SSFF 2D / iRMB 4D) → pnnx → ncnn 全链路数值保真,
树莓派上跑的就是论文里那个 mAP50=84.5% 的模型。
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import argparse
import glob
import os

import cv2
import numpy as np

SIZE = int(os.environ.get("V_SIZE", "640"))
HERE = os.path.dirname(os.path.abspath(__file__))
# 可用环境变量 V_DIR / V_SIZE 切换到其它输入分辨率的模型目录(如 416)
MODEL_DIR = os.environ.get("V_DIR", os.path.join(HERE, "model"))
PARAM = os.path.join(MODEL_DIR, "yolov11_mulberry.torchscript.ncnn.param")
BIN = os.path.join(MODEL_DIR, "yolov11_mulberry.torchscript.ncnn.bin")
TS = os.path.join(MODEL_DIR, "yolov11_mulberry.torchscript.pt")
OUT_ROOT = os.path.join(HERE, "_verify_out")
IMG_DIR = os.path.join(HERE, "images")


def preprocess(img_bgr, size=SIZE):
    """等比缩放 + 114 灰边填充, 返回 RGB uint8 图(letterbox 后)。

    注意: 模型期望 RGB 通道且输入范围 0~1(已用 ultralytics 原生 predict 逐项比对确认:
    用 BGR 或不归一化会退化成 conf=1.000 的错误输出)。归一化由调用方完成 ——
    torch 端除以 255, ncnn 端用 substract_mean_normalize。
    """
    h, w = img_bgr.shape[:2]
    r = size / max(h, w)
    nh, nw = int(round(h * r)), int(round(w * r))
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pl, pt = (size - nw) // 2, (size - nh) // 2
    canvas[pt:pt + nh, pl:pl + nw] = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(canvas[:, :, ::-1])  # BGR -> RGB


def images():
    return sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")))[:3]


def dump_torch():
    import torch
    outdir = os.path.join(OUT_ROOT, "torch")
    os.makedirs(outdir, exist_ok=True)
    model = torch.jit.load(TS)
    model.eval()
    for p in images():
        rgb = preprocess(cv2.imread(p))
        x = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)[None]
        with torch.no_grad():
            y = model(x)[0].numpy()
        np.save(os.path.join(outdir, os.path.basename(p) + ".npy"), y)
        print("torch:", os.path.basename(p), y.shape, flush=True)


def dump_ncnn():
    import ncnn
    outdir = os.path.join(OUT_ROOT, "ncnn")
    os.makedirs(outdir, exist_ok=True)
    net = ncnn.Net()
    net.opt.num_threads = 4
    net.opt.use_fp16_packed = False
    net.opt.use_fp16_storage = False
    net.opt.use_fp16_arithmetic = False
    assert net.load_param(PARAM) == 0 and net.load_model(BIN) == 0, "ncnn 模型加载失败"
    print("ncnn 模型加载成功", flush=True)
    for p in images():
        rgb = preprocess(cv2.imread(p))
        mat = ncnn.Mat.from_pixels(rgb, ncnn.Mat.PixelType.PIXEL_RGB, SIZE, SIZE)
        mat.substract_mean_normalize([0.0, 0.0, 0.0], [1 / 255.0, 1 / 255.0, 1 / 255.0])
        ex = net.create_extractor()
        ex.input("in0", mat)
        ret, out = ex.extract("out0")
        if ret != 0 or out.dims == 0:
            print(f"ncnn: {os.path.basename(p)} 推理失败(ret={ret})", flush=True)
            continue
        arr = np.array(out).reshape(out.h, out.w).copy()
        np.save(os.path.join(outdir, os.path.basename(p) + ".npy"), arr)
        print("ncnn:", os.path.basename(p), arr.shape, flush=True)


def compare():
    td, nd = os.path.join(OUT_ROOT, "torch"), os.path.join(OUT_ROOT, "ncnn")
    print(f"{'图片':<12}{'形状':<16}{'框坐标最大偏差(px)':<22}{'类别概率最大偏差'}")
    worst_box = worst_cls = 0.0
    for p in images():
        name = os.path.basename(p)
        yt = np.load(os.path.join(td, name + ".npy"))
        yn = np.load(os.path.join(nd, name + ".npy"))
        if yt.shape != yn.shape:
            print(f"{name:<12}形状不一致 torch={yt.shape} ncnn={yn.shape}")
            continue
        d = np.abs(yt - yn)
        bd, cd = float(d[:4].max()), float(d[4:].max())
        worst_box, worst_cls = max(worst_box, bd), max(worst_cls, cd)
        print(f"{name:<12}{str(yt.shape):<16}{bd:<22.4f}{cd:.5f}")
    print(f"\n最差: 框坐标 {worst_box:.4f} px, 类别概率 {worst_cls:.5f}")
    ok = worst_box < 0.5 and worst_cls < 1e-2
    print("判定:", "PASS —— 转换链路数值保真" if ok else "FAIL —— 偏差过大, 需排查")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["torch", "ncnn", "compare"])
    a = ap.parse_args()
    {"torch": dump_torch, "ncnn": dump_ncnn, "compare": compare}[a.mode]()
