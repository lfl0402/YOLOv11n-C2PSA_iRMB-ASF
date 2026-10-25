#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""桑叶病虫害检测系统 —— 网页界面(Gradio)。

启动(在 F:\\YOLOv11-2 目录下, 用 (pytorch) 环境):
    python app.py
浏览器会自动打开 http://127.0.0.1:7860

功能:
    单张检测 —— 上传/选择图片, 可调置信度与 NMS 阈值, 输出标注图 + 检测明细表 + 统计
    批量检测 —— 一次上传多张, 输出结果画廊与汇总明细
    说明     —— 模型信息、类别中英文对照、边缘部署实测指标

注意(重要):
    Gradio 的 gr.Image(type="pil") 给的是 PIL 图(RGB), 而 ultralytics 对 **numpy** 输入
    按 BGR 处理。若用 type="numpy" 直接把图喂给模型, 通道会反, 类别判定出错
    (实测 47.jpg 会被误判为黄斑叶甲)。因此这里统一用 type="pil" 传 PIL 对象。

同一份代码以后可直接放到树莓派上运行, 用局域网内手机/电脑浏览器访问。
"""
import glob
import os
import threading
import time
import warnings

warnings.filterwarnings("ignore")
# 本机 torch 与 timm 重复链接 OpenMP 运行库, 不设这个变量会报 OMP: Error #15
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from collections import Counter

# ⚠️ 导入顺序不能改: 必须先导入 ultralytics(会加载 torch), 再导入 gradio。
# 本机实测: 先 import gradio 再 import torch 会让 torch 的 c10.dll 初始化失败
# (OSError: [WinError 1114] 动态链接库(DLL)初始化例程失败)。
from ultralytics import YOLO

import gradio as gr
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ROOT, "weights", "best.pt")
EXAMPLE_DIR = os.path.join(ROOT, "assets", "samples")

# 类别中英文对照(顺序与 dataset/mulberry.yaml 一致)
CN = {
    "Anthracnose": "炭疽病",
    "Healthy": "健康叶片",
    "Inchworm": "尺蠖",
    "Bacterial Wilt": "青枯病",
    "Yellow Melyrid Leaf Beetle": "黄斑叶甲",
    "Willow Blue Leaf Beetle": "柳蓝叶甲",
}

_model = None


def get_model():
    """首次调用时加载模型, 之后复用, 避免每次推理都重新加载。"""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"找不到模型权重: {MODEL_PATH}")
        _model = YOLO(MODEL_PATH)
    return _model


def _rows(result):
    """把一张图的检测结果整理成表格行(按置信度降序)。"""
    rows = []
    if result.boxes is None:
        return rows
    for i in range(len(result.boxes)):
        name = result.names[int(result.boxes.cls[i])]
        x0, y0, x1, y1 = (int(v) for v in result.boxes.xyxy[i])
        rows.append([CN.get(name, name), name, round(float(result.boxes.conf[i]), 3), x0, y0, x1, y1])
    rows.sort(key=lambda r: -r[2])
    return rows


def _summary(rows):
    if not rows:
        return "未检出目标(可尝试调低置信度阈值)"
    c = Counter(r[0] for r in rows)
    return f"共检出 **{len(rows)}** 个目标:" + "、".join(f"{k} {v} 个" for k, v in c.most_common())


def detect_one(image, conf, iou):
    """单张检测 -> (标注图 RGB, 明细表, 统计文字)。"""
    if image is None:
        return None, [], "请先选择一张图片"
    result = get_model().predict(image, conf=conf, iou=iou, imgsz=640, verbose=False)[0]
    annotated = result.plot()[:, :, ::-1]        # plot() 输出 BGR, 转 RGB 供网页显示
    rows = _rows(result)
    return annotated, rows, _summary(rows)


def detect_batch(files, conf, iou):
    """批量检测 -> (画廊, 汇总明细, 统计文字)。"""
    if not files:
        return [], [], "请先选择图片"
    model = get_model()
    gallery, table, all_rows = [], [], []
    for f in files:
        path = f if isinstance(f, str) else getattr(f, "name", str(f))
        result = model.predict(path, conf=conf, iou=iou, imgsz=640, verbose=False)[0]
        rows = _rows(result)
        all_rows.extend(rows)
        name = os.path.basename(path)
        gallery.append((result.plot()[:, :, ::-1], f"{name} · {len(rows)} 个目标"))
        table.extend([[name] + r for r in rows])
    return gallery, table, _summary(all_rows)


# ---------------- 实时监控(USB 摄像头 / 演示模式) ----------------

SRC_CAM = "自动检测摄像头(USB)"
SRC_DEMO = "演示模式(循环播放验证集图片)"
SRC_CHOICES = [SRC_CAM, SRC_DEMO]

_mon = {"source": None, "kind": None, "cap": None, "last": None,
        "idx": 0, "n": 0, "t0": 0.0, "fps": 0.0,
        "probe_idx": None, "probe_t": 0.0}
_mon_lock = threading.Lock()


def _find_camera(max_index=5):
    """探测可用摄像头, 返回设备序号; 没有则 None。结果缓存 8 秒, 避免每帧重复探测。"""
    now = time.time()
    if _mon["probe_t"] and now - _mon["probe_t"] < 8:
        return _mon["probe_idx"]
    try:
        import cv2
    except Exception:
        return None
    # Windows 上优先 DSHOW, 比默认 MSMF 打开快很多; 其他平台用默认后端
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    found = None
    for i in range(max_index):
        cap = cv2.VideoCapture(i, backend)
        ok = cap.isOpened()
        if ok:
            ok, _ = cap.read()
        cap.release()
        if ok:
            found = i
            break
    _mon["probe_idx"], _mon["probe_t"] = found, now
    return found


def _close_camera():
    cap = _mon.get("cap")
    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass
    _mon.update(kind=None, cap=None)


def _open_source(source):
    """打开视频源 -> ('cam', VideoCapture) 或 ('demo', None)。"""
    if source != SRC_DEMO:
        idx = _find_camera()
        if idx is not None:
            import cv2
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                return "cam", cap
    return "demo", None


def _demo_image():
    if not os.path.isdir(EXAMPLE_DIR):
        return None
    imgs = sorted(glob.glob(os.path.join(EXAMPLE_DIR, "*.jpg")))
    if not imgs:
        return None
    path = imgs[_mon["idx"] % len(imgs)]
    _mon["idx"] += 1
    return path


def monitor_frame(source, conf, iou, running):
    """定时器每 0.5 s 触发: 抓一帧 -> 检测 -> 标注, 返回(标注图, 状态文字)。"""
    with _mon_lock:
        if not running:
            _close_camera()
            return _mon.get("last"), "⏸ 已暂停(勾选「开始监控」即开始)"

        need_open = _mon["source"] != source or _mon["kind"] is None
        # 之前没找到摄像头、现在可能插上了: 定期重试
        if not need_open and source != SRC_DEMO and _mon["kind"] == "demo":
            need_open = time.time() - _mon["probe_t"] > 8
        if need_open:
            _close_camera()
            _mon["source"] = source
            _mon["kind"], _mon["cap"] = _open_source(source)
            _mon["n"], _mon["t0"] = 0, time.time()

        if _mon["kind"] == "cam":
            ok, frame = _mon["cap"].read()
            if not ok:
                _close_camera()
                return _mon.get("last"), "⚠ 摄像头读取失败(设备可能被其他程序占用), 请稍后重试"
            pil = Image.fromarray(frame[:, :, ::-1])     # OpenCV BGR -> RGB
            tag = "USB 摄像头实时画面"
        else:
            path = _demo_image()
            if path is None:
                return _mon.get("last"), "未检测到摄像头, 且没有可用于演示的验证集图片"
            pil = Image.open(path).convert("RGB")
            tag = f"演示画面:{os.path.basename(path)}(未检测到摄像头)"

        result = get_model().predict(pil, conf=conf, iou=iou, imgsz=640, verbose=False)[0]
        n_obj = 0 if result.boxes is None else len(result.boxes)
        _mon["last"] = result.plot()[:, :, ::-1]          # BGR -> RGB
        _mon["n"] += 1
        dt = time.time() - _mon["t0"]
        if dt >= 3.0:
            _mon["fps"] = _mon["n"] / dt
            _mon["n"], _mon["t0"] = 0, time.time()

        status = f"**{tag}** · 本帧检出 {n_obj} 个目标"
        if _mon["fps"]:
            status += f" · 约 {_mon['fps']:.1f} FPS"
        return _mon["last"], status


HEADERS_ONE = ["类别", "英文名", "置信度", "x1", "y1", "x2", "y2"]
HEADERS_BATCH = ["图片"] + HEADERS_ONE
DT_ONE = ["str", "str", "number", "number", "number", "number", "number"]
DT_BATCH = ["str"] + DT_ONE


def build_ui():
    examples = []
    if os.path.isdir(EXAMPLE_DIR):
        for n in ("47.jpg", "48.jpg", "5.jpg", "57.jpg"):
            p = os.path.join(EXAMPLE_DIR, n)
            if os.path.exists(p):
                examples.append([p])

    with gr.Blocks(title="桑叶病虫害检测系统") as demo:
        gr.Markdown(
            "# 桑叶病虫害检测系统\n"
            "基于改进 YOLOv11n(C2PSA-iRMB + ASF 多尺度融合)的桑叶病虫害检测。"
            "验证集 mAP50 = **84.5%**,支持 6 类:炭疽病、健康叶片、尺蠖、青枯病、黄斑叶甲、柳蓝叶甲。"
        )

        with gr.Tab("实时监控"):
            with gr.Row():
                with gr.Column(scale=1):
                    src = gr.Dropdown(choices=SRC_CHOICES, value=SRC_CHOICES[0], label="视频源")
                    mconf = gr.Slider(0.05, 0.95, value=0.25, step=0.05,
                                      label="置信度阈值(调低可减少漏检)")
                    miou = gr.Slider(0.10, 0.90, value=0.45, step=0.05, label="NMS IoU 阈值")
                    run = gr.Checkbox(value=False, label="开始监控")
                    gr.Markdown(
                        "**说明**\n"
                        "- 插上 USB 摄像头(或直接用笔记本自带摄像头), 选「自动检测摄像头」即可\n"
                        "- 没检测到摄像头时自动回退到演示模式, 循环播放验证集图片\n"
                        "- 勾选「开始监控」后每 0.5 s 抓一帧检测, 实时显示框与类别\n"
                        "- 画面卡住时先取消勾选再重新勾选, 会重新打开摄像头"
                    )
                with gr.Column(scale=2):
                    live = gr.Image(label="实时画面(含检测框)", height=420)
                    mstat = gr.Markdown("等待开始…")
            timer = gr.Timer(0.5)
            timer.tick(monitor_frame, [src, mconf, miou, run], [live, mstat])

        with gr.Tab("单张检测"):
            with gr.Row():
                with gr.Column(scale=1):
                    img_in = gr.Image(label="待检测图片", type="pil", height=380)
                    conf = gr.Slider(0.05, 0.95, value=0.25, step=0.05,
                                     label="置信度阈值(调低可减少漏检)")
                    iou = gr.Slider(0.10, 0.90, value=0.45, step=0.05, label="NMS IoU 阈值")
                    btn = gr.Button("开始检测", variant="primary")
                    if examples:
                        gr.Examples(examples=examples, inputs=img_in, label="示例图片(点击试用)")
                with gr.Column(scale=1):
                    img_out = gr.Image(label="检测结果", height=380)
                    stat = gr.Markdown("等待检测…")
                    table = gr.Dataframe(headers=HEADERS_ONE, datatype=DT_ONE,
                                         interactive=False, label="检测明细")
            btn.click(detect_one, [img_in, conf, iou], [img_out, table, stat])

        with gr.Tab("批量检测"):
            with gr.Row():
                with gr.Column(scale=1):
                    files = gr.Files(label="选择多张图片(支持多选/拖入)", file_count="multiple",
                                     file_types=["image"], height=300)
                    bconf = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="置信度阈值")
                    biou = gr.Slider(0.10, 0.90, value=0.45, step=0.05, label="NMS IoU 阈值")
                    bbtn = gr.Button("批量检测", variant="primary")
                    bstat = gr.Markdown("等待检测…")
                with gr.Column(scale=2):
                    gallery = gr.Gallery(label="检测结果", columns=3, height=520)
            btable = gr.Dataframe(headers=HEADERS_BATCH, datatype=DT_BATCH,
                                  interactive=False, label="全部检测明细")
            bbtn.click(detect_batch, [files, bconf, biou], [gallery, btable, bstat])

        with gr.Tab("说明"):
            gr.Markdown(
                "## 模型\n"
                "- 结构:YOLOv11n + C2PSA-iRMB(Backbone 注意力)+ ASF 多尺度特征融合(Neck)\n"
                "- 权重:`weights/best.pt`\n"
                "- 验证集(645 张图像 / 1003 个标注实例):mAP50 = **84.5%**,召回率 79.2%\n\n"
                "## 类别对照\n"
                + "\n".join(f"- **{k}** —— {v}" for k, v in CN.items()) +
                "\n\n## 边缘部署实测(树莓派 4B)\n"
                "- FP32 @640×640:438 ms/帧(2.3 FPS),检测结果与 PC 端完全一致\n"
                "- FP32 @416×416:187 ms/帧(5.3 FPS),mAP50 降至 82.6%\n"
                "- INT8 量化:仅提速 1.14 倍且精度明显下降,未采用\n\n"
                "## 运行说明\n"
                "- 推理在**本机 CPU** 上完成(该模型体量小,实测 GPU 无明显收益)\n"
                "- 「实时监控」页支持 USB 摄像头/笔记本摄像头实时检测,"
                "没接摄像头时自动回退为验证集图片演示\n"
                "- 本页面为 Gradio 网页应用,同一份代码可直接部署到树莓派,"
                "由局域网内的手机或电脑浏览器访问"
            )
    return demo


if __name__ == "__main__":
    # 0.0.0.0: 同一局域网(手机热点/Wi-Fi)内的手机也能访问;
    # 若只想本机使用, 改回 "127.0.0.1"
    build_ui().launch(server_name="0.0.0.0", server_port=7860, inbrowser=True)
