#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLOv11n-C2PSA-iRMB-ASF(桑叶病虫害检测, 6 类) → torchscript → pnnx → NCNN 导出脚本

转换路线(注意 pnnx 的输入是 torchscript 的 .pt, 不是 ONNX):
    best.pt ──(torch.jit.trace)──> yolov11_mulberry.torchscript.pt
            ──(pnnx)──> .ncnn.param / .ncnn.bin
            ──(ncnnoptimize)──> 部署用 .param/.bin, 再按需 INT8 量化

用法(在仓库根目录 F:\\YOLOv11-2 下执行):
    python deploy/export.py                    # 默认使用 weights/best.pt(论文中 mAP50=84.5% 的模型)
    python deploy/export.py --weights weights/best.pt
    python deploy/export.py --img assets/samples/47.jpg   # 用真实图片做解码自检

输出头说明:
    脚本把模型输出包装成单一输出 out0, 形状 (1, 4+nc, 8400):
      - 前 4 行: 已解码的框坐标, 格式为 (cx, cy, w, h), 单位是 640x640 输入空间的像素
        (本 fork 的 Detect.decode_bboxes 用 xywh=True, 并乘了 stride, 已换算到像素)
      - 后 nc 行: sigmoid 后的类别概率(无 objectness)
      - 8400 = 80x80 + 40x40 + 20x20 三个尺度的网格, 按尺度从小到大拼接
    C++ 端(deploy/cpp)的后处理与本脚本的 decode_out() 完全一致: cx/cy/w/h -> xyxy -> NMS。

注意事项:
    - Neck 的 SSFF(ScalSeq)原版用 unsqueeze+cat 构造 5 维张量(Conv3d/MaxPool3d 路径),
      ncnn 不支持 5 维 permute/reshape。本脚本导出前会自动把它替换成数学等价的 2D 实现
      (ScalSeq2D, 见下), 并做原模型 vs 重写模型的输出一致性校验。
    - Windows 上 torch 与 timm 同时加载会报 OpenMP 冲突, 脚本开头已自动加 KMP_DUPLICATE_LIB_OK。
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # 必须在导入 torch/timm 之前设置

import argparse
import copy
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # 优先导入本地修改版 ultralytics(含 C2PSA_iRMB/ASF 自定义模块)

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScalSeq2D(nn.Module):
    """SSFF(ScalSeq)的 2D 等价实现, 供部署转换使用。

    原版通过 unsqueeze + cat 构造 (B, C, 3, H, W) 五维张量, 再用 Conv3d(1,1,1)、
    BatchNorm3d、MaxPool3d((3,1,1)) 处理 —— ncnn/pnnx 不支持 5 维 permute/reshape。
    由于 Conv3d 核为 1x1x1(不在"3"维上做混合), 三条尺度支路逐路卷积、逐路 BN+激活后
    取逐元素最大值, 与原版推理结果完全等价(推理期 BatchNorm3d 按通道仿射, 参数与
    BatchNorm2d 同形同义)。计算量与原版相同, 不增加开销。
    """

    def __init__(self, orig):
        super().__init__()
        C = orig.conv3d.weight.shape[0]
        self.conv0 = orig.conv0  # 复用原卷积(1x1)
        self.conv1 = orig.conv1
        self.conv2 = orig.conv2
        self.conv3d = nn.Conv2d(C, C, 1)  # 原 Conv3d(C, C, (1,1,1))
        self.bn = nn.BatchNorm2d(C)       # 原 BatchNorm3d(C)
        self.act = nn.LeakyReLU(0.1)
        with torch.no_grad():
            self.conv3d.weight.copy_(orig.conv3d.weight.view(C, C, 1, 1))
            self.conv3d.bias.copy_(orig.conv3d.bias)
            self.bn.load_state_dict(orig.bn.state_dict())
        # ultralytics 在 parse_model 阶段会给模块挂 f/i/type 属性, 前向时会用到, 必须带上
        for attr in ("f", "i", "type"):
            if hasattr(orig, attr):
                setattr(self, attr, getattr(orig, attr))
        # 新建的子模块默认 training=True(BatchNorm2d 会用批次统计而非 running stats),
        # 本模块只用于部署导出, 强制 eval
        self.eval()

    def forward(self, x):
        p3, p4, p5 = x[0], x[1], x[2]
        p3 = self.conv0(p3)
        p4 = F.interpolate(self.conv1(p4), p3.size()[2:], mode="nearest")
        p5 = F.interpolate(self.conv2(p5), p3.size()[2:], mode="nearest")
        y3 = self.act(self.bn(self.conv3d(p3)))
        y4 = self.act(self.bn(self.conv3d(p4)))
        y5 = self.act(self.bn(self.conv3d(p5)))
        return torch.maximum(torch.maximum(y3, y4), y5)


def replace_scalseq_with_2d(net):
    """把模型里所有 ScalSeq(SSFF)模块替换为 2D 等价实现, 返回替换数量。"""
    names = [name for name, m in net.named_modules() if type(m).__name__ == "ScalSeq"]
    for name in names:
        parent = net
        leaf = name
        if "." in name:
            parts = name.split(".")
            for p in parts[:-1]:
                parent = getattr(parent, p)
            leaf = parts[-1]
        orig = getattr(parent, leaf)
        setattr(parent, leaf, ScalSeq2D(orig))
    return len(names)


class iRMB4D(nn.Module):
    """iRMB 的部署等价实现: 窗口注意力全部用 ≤4D 的 view/permute/matmul 表达。

    原版用 einops.rearrange 构造 5/6 维张量(qk b heads (hw) dim_head 等),
    pnnx/ncnn 不支持 5 维 permute/reshape。本实现:
      - 窗口切分/合并用 "4D reshape + permute" 完成(逐点卷积与注意力在
        窗口内 (行,列) 转置布局下不变, 合并时用同样的转置还原);
      - 多头拆分利用 C = heads*dim_head 的整除性做合法 4D reshape;
      - qkv 的 1x1 逐点卷积移到"窗口划分之前"直接复用原模块(1x1 卷积与窗口划分
        可交换), 从而完全避免分组卷积 —— ncnn 的分组卷积在 4D 布局上会运行失败。
    数学上与原版完全等价。仅支持本模型实际使用的配置: attn_s=True, attn_pre=False,
    exp_ratio=1.0, v_group=False。
    """

    def __init__(self, orig, feat_hw):
        super().__init__()
        assert orig.attn_s and not orig.attn_pre, "仅支持 attn_s=True/attn_pre=False 配置"
        self.norm = orig.norm
        self.qk = orig.qk          # 1x1 逐点卷积 + Identity 激活
        self.v = orig.v            # 1x1 逐点卷积 + ReLU
        self.dim_head = orig.dim_head
        self.window_size = orig.window_size
        self.num_head = orig.num_head
        self.scale = orig.scale
        self.has_skip = orig.has_skip
        self.attn_drop = orig.attn_drop
        self.conv_local = orig.conv_local
        self.se = orig.se
        self.proj_drop = orig.proj_drop
        self.proj = orig.proj
        self.drop_path = orig.drop_path
        for attr in ("f", "i", "type"):
            if hasattr(orig, attr):
                setattr(self, attr, getattr(orig, attr))
        # 窗口网格(静态): 该模块位于 P5 尺度, feat_hw = 输入尺寸 // 32
        ws = orig.window_size
        if ws <= 0:
            n_windows = 1
        else:
            pad = (ws - feat_hw % ws) % ws
            n = (feat_hw + pad) // ws
            n_windows = n * n
        self._N = n_windows
        self.eval()

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        B, C, H, W = x.shape
        ws = self.window_size
        if ws <= 0:
            h1, w1, n1, n2 = H, W, 1, 1
            pad_r = pad_b = 0
        else:
            pad_r = (ws - W % ws) % ws
            pad_b = (ws - H % ws) % ws
            h1 = w1 = ws
            n1 = (H + pad_b) // ws
            n2 = (W + pad_r) // ws
        N = n1 * n2
        assert n1 == n2 and N == self._N, f"窗口配置不支持: n1={n1} n2={n2} vs {self._N}"
        heads, dh = self.num_head, self.dim_head
        hw = h1 * w1

        # ---- 1x1 逐点卷积与窗口划分可交换: 先在整图上做 qkv 卷积, 再填充 + 划分 ----
        pad = (0, pad_r, 0, pad_b)
        qk = self.qk(x)                                    # (B, 2C, H, W)
        q = F.pad(qk[:, :C], pad)
        k = F.pad(qk[:, C:], pad)
        v = F.pad(self.v(x), pad)                          # ReLU 已包含在 self.v 内

        # ---- 窗口划分: 原版 einops 的 (h1 n1) 分解是"跨步采样"(窗口 (i,j) 取
        #      行%n1==i、列%n2==j 的像素), 等价于 pixel_unshuffle, pnnx 转为 ncnn
        #      原生 Reorg 层。通道排布 c*N+g → g*C+c 用 4D permute 完成 ----
        def part(t):
            xs = F.pixel_unshuffle(t, n1)                  # (B, C*N, h1, w1), 通道 = c*N+g
            return xs.view(B, C, N, hw).permute(0, 2, 1, 3).reshape(B, N * C, h1, w1)

        qw = part(q).view(B, N * heads, dh, hw)
        kw = part(k).view(B, N * heads, dh, hw)
        vw = part(v).view(B, N * heads, dh, hw)

        # ---- 注意力: 批量维 = 窗口×头 ----
        attn = (qw.permute(0, 1, 3, 2) @ kw) * self.scale   # (B, N*heads, hw, hw)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_spa = (attn @ vw.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)   # (B, N*heads, dh, hw)
        x_spa = x_spa.reshape(B, N * C, h1, w1)

        # ---- 窗口合并: 与划分互逆(排布还原 + pixel_shuffle → ncnn PixelShuffle) ----
        xm = x_spa.view(B, N, C, hw).permute(0, 2, 1, 3).reshape(B, C * N, h1, w1)
        xm = F.pixel_shuffle(xm, n1)
        if pad_r > 0 or pad_b > 0:
            xm = xm[:, :, :H, :W].contiguous()

        # ---- 与原版相同的尾部 ----
        x = xm + self.se(self.conv_local(xm)) if self.has_skip else self.se(self.conv_local(xm))
        x = self.proj_drop(x)
        x = self.proj(x)
        x = (shortcut + self.drop_path(x)) if self.has_skip else x
        return x


def replace_irmb_with_4d(net, feat_hw):
    """把模型里所有 iRMB 模块替换为 4D 等价实现, 返回替换数量。"""
    names = [name for name, m in net.named_modules() if type(m).__name__ == "iRMB"]
    for name in names:
        parent = net
        leaf = name
        if "." in name:
            parts = name.split(".")
            for p in parts[:-1]:
                parent = getattr(parent, p)
            leaf = parts[-1]
        orig = getattr(parent, leaf)
        setattr(parent, leaf, iRMB4D(orig, feat_hw))
    return len(names)


class DeployWrapper(nn.Module):
    """把 ultralytics 模型的 (y, x) 元组输出裁剪成单一输出 y(1, 4+nc, 8400, 已解码)。"""

    def __init__(self, core):
        super().__init__()
        self.core = core

    def forward(self, x):
        y = self.core(x)
        if isinstance(y, (tuple, list)):
            y = y[0]
        return y


def letterbox(img_bgr, size=640, color=(114, 114, 114)):
    """等比缩放 + 灰边填充, 返回 RGB uint8 图。

    与 C++ 端(deploy/cpp/yolov11.cpp)严格一致: 缩放 → 114 灰边填充 → BGR 转 RGB,
    随后输入归一化到 0~1。模型期望 **RGB 通道 + 0~1 范围**(已用 ultralytics 原生
    predict 逐项比对确认; 若用 BGR 或不做归一化, 输出会退化成 conf=1.000 的错误结果)。
    """
    h, w = img_bgr.shape[:2]
    r = size / max(h, w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), color, dtype=np.uint8)
    pad_l = (size - nw) // 2
    pad_t = (size - nh) // 2
    canvas[pad_t:pad_t + nh, pad_l:pad_l + nw] = resized
    return np.ascontiguousarray(canvas[:, :, ::-1]), r, pad_l, pad_t


def decode_out(out, nc, conf, nms=0.45):
    """out: (1, 4+nc, 8400) tensor → [(cls, score, x0, y0, x1, y1)] in 640 空间。

    与 C++ 端 deploy/cpp/yolov11.cpp 的后处理逻辑一一对应:
    框为 (cx, cy, w, h) → 转 xyxy → 按分数降序贪心 NMS。
    """
    a = out.detach().cpu().numpy()[0]
    cands = []
    for j in range(a.shape[1]):
        scores = a[4:, j]
        k = int(scores.argmax())
        s = float(scores[k])
        if s < conf:
            continue
        cx, cy, w, h = (float(v) for v in a[:4, j])
        cands.append((k, s, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    cands.sort(key=lambda b: -b[1])

    def iou(p, q):
        ix0, iy0 = max(p[2], q[2]), max(p[3], q[3])
        ix1, iy1 = min(p[4], q[4]), min(p[5], q[5])
        inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
        return inter / max(1e-9, (p[4] - p[2]) * (p[5] - p[3]) + (q[4] - q[2]) * (q[5] - q[3]) - inter)

    keep = []
    for c in cands:
        if all(iou(c, k) <= nms for k in keep):
            keep.append(c)
    return keep


def main():
    ap = argparse.ArgumentParser(description="桑叶病虫害 YOLOv11n 改进模型 → torchscript(pnnx 输入)")
    ap.add_argument("--weights", default=str(ROOT / "weights" / "best.pt"))
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--img", default="", help="可选: 真实图片路径, 用于导出后解码自检")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "model"))
    ap.add_argument("--run-pnnx", action="store_true", help="若系统 PATH 中有 pnnx, 转换完成后自动执行")
    args = ap.parse_args()

    from ultralytics import YOLO  # 延迟导入: 保证 ROOT 已加入 sys.path

    print(f"[1/4] 加载模型: {args.weights}")
    model = YOLO(args.weights)
    net_orig = model.model
    nc = int(net_orig.model[-1].nc)
    names = model.names
    print(f"      类别数 nc={nc}: {names}")

    net_orig.eval()
    for p in net_orig.parameters():
        p.requires_grad_(False)

    # 部署友好重写: SSFF(ScalSeq)的 5 维张量路径换成数学等价的 2D 实现(ncnn 不支持 5 维)
    net = copy.deepcopy(net_orig)
    n_rep = replace_scalseq_with_2d(net)
    # iRMB 窗口注意力(einops 5/6 维张量)换成 4D 等价实现
    n_irmb = replace_irmb_with_4d(net, args.size // 32)
    net.eval()  # 重写引入的新子模块要确保处于 eval(内部已强制, 这里再兜底一次)
    print(f"      已把 {n_rep} 个 ScalSeq(SSFF)替换为 2D 等价实现, {n_irmb} 个 iRMB 替换为 4D 等价实现")

    print(f"[2/4] torch.jit.trace(输入 1x3x{args.size}x{args.size}, CPU FP32)")
    torch.manual_seed(0)  # 固定随机输入, 使 trace 与后续 pnnx 常量折叠可复现(层数/FLOPs 稳定)
    x = torch.rand(1, 3, args.size, args.size)
    wrapper = DeployWrapper(net)
    wrapper_orig = DeployWrapper(net_orig)
    with torch.no_grad():
        # 先做一次真实前向: 让 Detect 头把 anchors/strides/shape 缓存下来。
        # 否则 "首次调用才计算 anchors" 的分支会让两次 trace 构图不一致而报错。
        wrapper(x)
        wrapper_orig(x)
        # 等价性校验: 重写后的模型与原模型输出必须一致。
        # 阈值取 1e-3: 输出前 4 行是 0~640 的像素坐标, 网络约 200 层 float32 累积误差
        # 在 1e-4 量级(相对误差 ~1e-7), 属数值噪声; 下方自检的检测框坐标应完全一致。
        y_new = wrapper(x)
        y_old = wrapper_orig(x)
        eq_diff = (y_new - y_old).abs().max().item()
        print(f"      等价性校验: max_abs_diff = {eq_diff:.3e} (期望 < 1e-3, 像素尺度相对误差)")
        if eq_diff > 1e-3:
            raise RuntimeError("重写版与原模型输出不一致, 请检查 ScalSeq2D / iRMB4D 的等价实现")
        traced = torch.jit.trace(wrapper, x, strict=False)
        eager_out = wrapper(x)
        traced_out = traced(x)

    diff = (eager_out - traced_out).abs().max().item()
    print(f"      trace 一致性校验: max_abs_diff = {diff:.3e} (期望 < 1e-3)")
    if diff > 1e-3:
        raise RuntimeError("trace 产物与原模型输出差异过大, 请检查自定义模块是否含动态 shape 分支")
    assert traced_out.shape[0] == 1 and traced_out.shape[1] == 4 + nc, \
        f"输出形状异常: {tuple(traced_out.shape)}, 期望 (1, {4 + nc}, 8400)"
    print(f"      输出形状: {tuple(traced_out.shape)} = (1, 4+{nc}, 8400)  ✓")

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    ts_path = outdir / "yolov11_mulberry.torchscript.pt"
    traced.save(str(ts_path))
    print(f"[3/4] 已保存 torchscript: {ts_path}")

    if args.img:
        print(f"[4/4] 真实图片解码自检: {args.img}")
        img0 = cv2.imread(args.img)
        if img0 is None:
            raise FileNotFoundError(f"读图失败: {args.img}")
        canvas, r, pad_l, pad_t = letterbox(img0, args.size)
        blob = torch.from_numpy(canvas.astype(np.float32) / 255.0).permute(2, 0, 1)[None]  # RGB, 0~1
        with torch.no_grad():
            out = traced(blob)
        boxes = decode_out(out, nc, args.conf)
        print(f"      检出 {len(boxes)} 个目标(NMS 后, conf≥{args.conf}):")
        for k, s, x0, y0, x1, y1 in boxes[:20]:
            ox0, oy0 = (x0 - pad_l) / r, (y0 - pad_t) / r
            ox1, oy1 = (x1 - pad_l) / r, (y1 - pad_t) / r
            print(f"      {str(names[k]):35s} conf={s:.3f}  box=({ox0:.0f},{oy0:.0f},{ox1:.0f},{oy1:.0f})")
        if not boxes:
            print("      (无目标, 可尝试 --conf 0.1 或换一张包含明显病虫害目标的图)")
    else:
        print("[4/4] 跳过图片自检(可用 --img 指定一张图验证解码是否正确)")

    print()
    print("下一步 —— pnnx 转换(两步都必须照做, 否则树莓派上会加载/运行失败):")
    print("  1) 安装与目标 ncnn 版本一致的 pnnx:  pip install -U pnnx")
    print("     (注意: 不要用 ultralytics 目录下自带的 pnnx.exe, 它与新版 ncnn 的")
    print("      Reshape 参数约定不一致, 会导致 ncnn 运行时报错或取不到输出)")
    print(f"  2) pnnx {Path(ts_path).name} inputshape=[1,3,{args.size},{args.size}] fp16=0")
    print("     fp16=0 必须加: 树莓派 4B(Cortex-A72)没有 FP16/INT8 点积指令,")
    print("     pnnx 默认 fp16=1 会把权重压成半精度, 实测会给输出带来明显偏差")
    print("     (框坐标最大偏差 12px → 加 fp16=0 后降到 0.001px)。")
    print("  # 产物: yolov11_mulberry.torchscript.ncnn.param / .ncnn.bin")
    print("  # pnnx 已按 optlevel=2 做过图优化, FP32 部署可不必再跑 ncnnoptimize。")
    print()
    print("验收 —— 在 PC 上先比对 PyTorch 与 ncnn 的输出(需要 pip install ncnn):")
    print("  python verify_ncnn.py --mode torch")
    print("  python verify_ncnn.py --mode ncnn")
    print("  python verify_ncnn.py --mode compare   # 期望框坐标偏差 < 0.5px")
    if args.run_pnnx:
        subprocess.run(["pnnx", str(ts_path), f"inputshape=[1,3,{args.size},{args.size}]"], check=True)


if __name__ == "__main__":
    main()
