#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""桑叶病虫害检测系统 —— 树莓派端网页界面(NCNN 推理, 不依赖 PyTorch)。

在树莓派上运行:
    cd ~/mulberry
    ./venv/bin/python pi_app.py                 # 启动网页, 端口 7860
    ./venv/bin/python pi_app.py --selftest 图片  # 不开网页, 只跑一张图自检
    ./venv/bin/python pi_app.py --port 8080

本机(局域网内电脑/手机)浏览器访问:
    http://<树莓派IP>:7860

推理用 deploy 阶段转换好的 NCNN 模型(FP32, 640×640), 前处理与 PC 端严格一致:
等比缩放 → 灰边 114 填充 → RGB → 归一化到 0~1。
"""
import argparse
import glob
import os
import sys
import time
import warnings
from collections import Counter

warnings.filterwarnings("ignore")

import numpy as np
import ncnn
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get("MULBERRY_MODEL_DIR", os.path.join(ROOT, "model"))
PARAM = os.path.join(MODEL_DIR, "yolov11_mulberry.torchscript.ncnn.param")
BIN = os.path.join(MODEL_DIR, "yolov11_mulberry.torchscript.ncnn.bin")
SIZE = 640

NAMES = ["Anthracnose", "Healthy", "Inchworm", "Bacterial Wilt",
         "Yellow Melyrid Leaf Beetle", "Willow Blue Leaf Beetle"]
CN = dict(zip(NAMES, ["炭疽病", "健康叶片", "尺蠖", "青枯病", "黄斑叶甲", "柳蓝叶甲"]))
COLORS = [(255, 56, 56), (56, 200, 56), (255, 149, 0), (0, 170, 255), (200, 60, 255), (255, 220, 0)]

_net = None


def get_net():
    """加载 NCNN 模型(只加载一次, 之后复用)。"""
    global _net
    if _net is None:
        if not (os.path.exists(PARAM) and os.path.exists(BIN)):
            raise FileNotFoundError(f"找不到 NCNN 模型: {PARAM} / {BIN}")
        net = ncnn.Net()
        net.opt.num_threads = 4
        net.opt.use_vulkan_compute = False
        # 树莓派 4B(Cortex-A72, ARMv8.0)没有 FP16/INT8 点积指令
        net.opt.use_fp16_packed = False
        net.opt.use_fp16_storage = False
        net.opt.use_fp16_arithmetic = False
        assert net.load_param(PARAM) == 0, "load_param 失败"
        assert net.load_model(BIN) == 0, "load_model 失败"
        _net = net
    return _net


def preprocess(im):
    """等比缩放 + 114 灰边填充 -> RGB uint8。返回 (数组, 缩放比, 左填充, 上填充)。"""
    w, h = im.size
    r = SIZE / max(w, h)
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = im.convert("RGB").resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (SIZE, SIZE), (114, 114, 114))
    pl, pt = (SIZE - nw) // 2, (SIZE - nh) // 2
    canvas.paste(resized, (pl, pt))
    return np.asarray(canvas), r, pl, pt


def detect(im, conf=0.25, iou=0.45):
    """对 PIL 图做检测。返回 [(类别序号, 置信度, x0, y0, x1, y1)](原图坐标, 已 NMS)。"""
    rgb, r, pl, pt = preprocess(im)
    net = get_net()
    mat = ncnn.Mat.from_pixels(rgb, ncnn.Mat.PixelType.PIXEL_RGB, SIZE, SIZE)
    mat.substract_mean_normalize([0.0, 0.0, 0.0], [1 / 255.0, 1 / 255.0, 1 / 255.0])
    ex = net.create_extractor()
    ex.input("in0", mat)
    ret, out = ex.extract("out0")
    if ret != 0 or out.dims == 0:
        raise RuntimeError(f"NCNN 推理失败(ret={ret})")
    y = np.array(out).reshape(out.h, out.w)

    cands = []
    for j in range(y.shape[1]):
        s = y[4:, j]
        k = int(s.argmax())
        if s[k] < conf:
            continue
        cx, cy, bw, bh = y[:4, j]
        cands.append((k, float(s[k]),
                      (cx - bw / 2 - pl) / r, (cy - bh / 2 - pt) / r,
                      (cx + bw / 2 - pl) / r, (cy + bh / 2 - pt) / r))
    cands.sort(key=lambda c: -c[1])

    def _iou(p, q):
        ix0, iy0 = max(p[2], q[2]), max(p[3], q[3])
        ix1, iy1 = min(p[4], q[4]), min(p[5], q[5])
        inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
        return inter / max(1e-9, (p[4] - p[2]) * (p[5] - p[3]) + (q[4] - q[2]) * (q[5] - q[3]) - inter)

    keep = []
    for c in cands:
        if all(_iou(c, k) <= iou for k in keep):
            keep.append(c)
    return keep


def _font():
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, 16)
            except Exception:
                pass
    return ImageFont.load_default()


def draw(im, boxes):
    """在图上画框(标签用英文, 因为树莓派默认字体不含中文)。"""
    out = im.convert("RGB").copy()
    d = ImageDraw.Draw(out)
    f = _font()
    for k, s, x0, y0, x1, y1 in boxes:
        color = COLORS[k % len(COLORS)]
        d.rectangle([x0, y0, x1, y1], outline=color, width=3)
        label = f"{NAMES[k]} {s:.2f}"
        tb = d.textbbox((x0, y0), label, font=f)
        ty = max(0, y0 - (tb[3] - tb[1]) - 4)
        d.rectangle([x0, ty, x0 + (tb[2] - tb[0]) + 6, ty + (tb[3] - tb[1]) + 4], fill=color)
        d.text((x0 + 3, ty + 2), label, fill=(255, 255, 255), font=f)
    return out


def rows_of(boxes):
    rows = sorted([[CN[NAMES[k]], NAMES[k], round(s, 3), int(x0), int(y0), int(x1), int(y1)]
                   for k, s, x0, y0, x1, y1 in boxes], key=lambda r: -r[2])
    return rows


def summary_of(rows):
    if not rows:
        return "未检出目标(可尝试调低置信度阈值)"
    c = Counter(r[0] for r in rows)
    return f"共检出 **{len(rows)}** 个目标:" + "、".join(f"{k} {v} 个" for k, v in c.most_common())


# ---------------- 实时监控 ----------------

SRC_CAM = "自动检测摄像头(USB)"
SRC_DEMO = "演示模式(循环播放验证集图片)"
SRC_CHOICES = [SRC_CAM, SRC_DEMO]

# 手机"添加到主屏幕"所需的元信息: iOS 用 apple-* 系列, Android 用 mobile-web-app-capable。
# 这样在手机浏览器里"添加到主屏幕"后会以独立窗口打开, 看起来就是一个 App。
PWA_HEAD = """
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="桑叶检测">
<meta name="mobile-web-app-capable" content="yes">
<meta name="theme-color" content="#2e7d32">
"""

MON_IMAGES = sorted(glob.glob(os.path.join(ROOT, "images", "*.jpg"))) or \
             [os.path.join(ROOT, "images", "47.jpg")]

_mon = {"kind": None, "cap": None, "idx": 0, "last": None, "n": 0, "t0": 0.0, "fps": 0.0, "source": None}


def _find_camera(max_index=4):
    """探测可用的 USB/V4L2 摄像头, 返回设备序号; 没有则 None。"""
    try:
        import cv2
    except Exception:
        return None
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ok, _ = cap.read()
            cap.release()
            if ok:
                return i
        cap.release()
    return None


def _close_monitor():
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


def monitor_frame(source, conf, iou, running):
    """定时器每次触发抓一帧并检测, 返回(标注图, 状态文字)。"""
    if not running:
        return _mon.get("last"), "⏸ 已暂停(勾选左侧「开始监控」即开始)"

    if _mon["source"] != source:                 # 切换了视频源 -> 重新打开
        _close_monitor()
        _mon["source"] = source
        _mon["kind"], _mon["cap"] = _open_source(source)
        _mon["n"], _mon["t0"] = 0, time.time()

    if _mon["kind"] == "cam":
        ok, frame = _mon["cap"].read()
        if not ok:
            return _mon.get("last"), "⚠ 摄像头读取失败(设备可能被占用)"
        pil = Image.fromarray(frame[:, :, ::-1])       # OpenCV 是 BGR -> 转 RGB
        tag = "USB 摄像头实时画面"
    else:
        path = MON_IMAGES[_mon["idx"] % len(MON_IMAGES)]
        _mon["idx"] += 1
        pil = Image.open(path).convert("RGB")
        tag = f"演示画面:{os.path.basename(path)}(未检测到摄像头)"

    boxes = detect(pil, conf, iou)
    ann = draw(pil, boxes)
    _mon["last"] = ann
    _mon["n"] += 1
    dt = time.time() - _mon["t0"]
    if dt >= 3.0:
        _mon["fps"] = _mon["n"] / dt
        _mon["n"], _mon["t0"] = 0, time.time()
    fps = _mon["fps"]
    return ann, f"**{tag}** · 本帧检出 {len(boxes)} 个目标" + (f" · 约 {fps:.1f} FPS" if fps else "")


# ---------------- 界面主题与样式 ----------------

# 农业绿主题: 统一主色、圆角与层次, 让默认的框架观感更像一个"设计过的系统"
THEME_CSS = """
.gradio-container { max-width: 1180px !important; margin: 0 auto !important; }
footer { display: none !important; }

/* 顶部标题横幅 */
#app-header {
    background: linear-gradient(135deg, #2e7d32 0%, #66bb6a 100%);
    color: #fff; padding: 20px 24px; border-radius: 14px;
    margin-bottom: 16px; box-shadow: 0 3px 12px rgba(46,125,50,.22);
}
#app-header h1 { margin: 0; font-size: 23px; font-weight: 700; color: #fff !important; }
#app-header .sub { margin: 8px 0 0; font-size: 14px; opacity: .93; line-height: 1.6; }
#app-header .tags { margin-top: 10px; }
#app-header .tag {
    display: inline-block; background: rgba(255,255,255,.22); color: #fff;
    border-radius: 20px; padding: 3px 12px; font-size: 12.5px; margin-right: 8px;
}

/* 卡片化的分页与面板 */
.tab-nav button { font-size: 15px !important; font-weight: 600 !important; }
.block, .form { border-radius: 12px !important; }
.gradio-container .prose h2, .gradio-container .prose h3 { color: #1b5e20; }

/* 检测明细表: 数字用等宽字体, 便于核对 */
#detail-table table td { font-variant-numeric: tabular-nums; }
"""

HEADER_HTML = """
<div id="app-header">
  <h1>桑叶病虫害检测系统</h1>
  <div class="sub">
    树莓派 4B 边缘端 · NCNN 就地推理,不依赖云端<br>
    改进 YOLOv11n(C2PSA-iRMB + ASF 多尺度融合)
  </div>
  <div class="tags">
    <span class="tag">验证集 mAP50 84.5%</span>
    <span class="tag">6 类病虫害</span>
    <span class="tag">单帧约 0.6 s</span>
  </div>
</div>
"""


# ---------------- 网页界面 ----------------

def detect_one(image, conf, iou):
    if image is None:
        return None, [], "请先选择一张图片"
    t0 = time.time()
    boxes = detect(image, conf, iou)
    ms = (time.time() - t0) * 1000
    rows = rows_of(boxes)
    return draw(image, boxes), rows, summary_of(rows) + f"\n\n单帧耗时 **{ms:.0f} ms**(树莓派 4B, 4 线程)"


def detect_batch(files, conf, iou):
    if not files:
        return [], [], "请先选择图片"
    gallery, table, all_rows, total_ms = [], [], [], 0.0
    for f in files:
        path = f if isinstance(f, str) else getattr(f, "name", str(f))
        im = Image.open(path)
        t0 = time.time()
        boxes = detect(im, conf, iou)
        total_ms += (time.time() - t0) * 1000
        rows = rows_of(boxes)
        all_rows.extend(rows)
        name = os.path.basename(path)
        gallery.append((draw(im, boxes), f"{name} · {len(rows)} 个目标"))
        table.extend([[name] + r for r in rows])
    n = max(1, len(files))
    return gallery, table, summary_of(all_rows) + f"\n\n平均单帧 **{total_ms / n:.0f} ms**"


HEADERS_ONE = ["类别", "英文名", "置信度", "x1", "y1", "x2", "y2"]
HEADERS_BATCH = ["图片"] + HEADERS_ONE
DT_ONE = ["str", "str", "number", "number", "number", "number", "number"]
DT_BATCH = ["str"] + DT_ONE


def build_ui():
    import gradio as gr

    theme = gr.themes.Soft(primary_hue="green", secondary_hue="green", neutral_hue="slate")

    with gr.Blocks(title="桑叶病虫害检测系统", theme=theme, css=THEME_CSS, head=PWA_HEAD) as demo:
        gr.HTML(HEADER_HTML)
        with gr.Tab("实时监控"):
            with gr.Row():
                with gr.Column(scale=1):
                    src = gr.Dropdown(choices=SRC_CHOICES, value=SRC_CHOICES[0], label="视频源")
                    mconf = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="置信度阈值")
                    miou = gr.Slider(0.10, 0.90, value=0.45, step=0.05, label="NMS IoU 阈值")
                    run = gr.Checkbox(value=False, label="开始监控")
                    gr.Markdown(
                        "**说明**\n"
                        "- 插上 USB 摄像头后选「自动检测摄像头」即为实时画面\n"
                        "- 当前未接摄像头, 会自动使用演示模式(循环播放验证集图片)\n"
                        "- 树莓派 4B 单帧推理约 0.6 s, 因此画面刷新约 1~2 FPS, 属正常\n"
                        "- 手机浏览器菜单 →「添加到主屏幕」, 即可像 App 一样使用"
                    )
                with gr.Column(scale=2):
                    live = gr.Image(label="实时画面(含检测框)", height=420)
                    mstat = gr.Markdown("等待开始…")
            timer = gr.Timer(0.5)
            timer.tick(monitor_frame, [src, mconf, miou, run], [live, mstat])

        with gr.Tab("单张检测"):
            with gr.Row():
                with gr.Column(scale=1):
                    img_in = gr.Image(label="待检测图片", type="pil", height=360)
                    conf = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="置信度阈值")
                    iou = gr.Slider(0.10, 0.90, value=0.45, step=0.05, label="NMS IoU 阈值")
                    btn = gr.Button("开始检测", variant="primary")
                with gr.Column(scale=1):
                    img_out = gr.Image(label="检测结果", height=360)
                    stat = gr.Markdown("等待检测…")
                    table = gr.Dataframe(headers=HEADERS_ONE, datatype=DT_ONE,
                                         interactive=False, label="检测明细")
            btn.click(detect_one, [img_in, conf, iou], [img_out, table, stat])

        with gr.Tab("批量检测"):
            with gr.Row():
                with gr.Column(scale=1):
                    files = gr.Files(label="选择多张图片", file_count="multiple",
                                     file_types=["image"], height=280)
                    bconf = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="置信度阈值")
                    biou = gr.Slider(0.10, 0.90, value=0.45, step=0.05, label="NMS IoU 阈值")
                    bbtn = gr.Button("批量检测", variant="primary")
                    bstat = gr.Markdown("等待检测…")
                with gr.Column(scale=2):
                    gallery = gr.Gallery(label="检测结果", columns=3, height=480)
            btable = gr.Dataframe(headers=HEADERS_BATCH, datatype=DT_BATCH,
                                  interactive=False, label="全部检测明细")
            bbtn.click(detect_batch, [files, bconf, biou], [gallery, btable, bstat])

        with gr.Tab("说明"):
            gr.Markdown(
                "## 边缘端配置\n"
                "- 设备:树莓派 4B(4 GB,Cortex-A72 四核)\n"
                "- 推理引擎:NCNN(源码编译,与模型转换用的 pnnx 版本配套)\n"
                "- 模型:FP32 640×640,无 FP16/INT8 加速(该 CPU 无相应指令)\n"
                "- 实测:438 ms/帧(约 2.3 FPS),与 PC 端检测结果逐项一致\n\n"
                "## 类别对照\n" + "\n".join(f"- **{k}** —— {v}" for k, v in CN.items())
            )
    return demo


def selftest(path):
    im = Image.open(path)
    t0 = time.time()
    boxes = detect(im, 0.25, 0.45)
    ms = (time.time() - t0) * 1000
    print(f"图片: {path}  尺寸: {im.size}")
    print(f"检出 {len(boxes)} 个目标, 耗时 {ms:.0f} ms:")
    for k, s, x0, y0, x1, y1 in boxes:
        print(f"  {NAMES[k]:28s} conf={s:.3f}  ({x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f})")
    out = os.path.join(ROOT, "out_pi_app.jpg")
    draw(im, boxes).save(out, quality=92)
    print("标注图:", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--selftest", default="")
    ap.add_argument("--auth", default="",
                    help="形如 用户名:密码。一旦把服务暴露到公网(内网穿透/端口映射), 必须设置, "
                         "否则任何人拿到网址都能使用并上传文件")
    a = ap.parse_args()
    if a.selftest:
        selftest(a.selftest)
    else:
        launch_kw = {}
        if a.auth and ":" in a.auth:
            user, pwd = a.auth.split(":", 1)
            launch_kw["auth"] = (user, pwd)
            print(f"[已启用访问认证] 用户名: {user}")
        else:
            print("[提示] 当前无访问认证, 仅建议在局域网内使用; "
                  "如需公网访问请加 --auth 用户名:密码")
        build_ui().launch(server_name=a.host, server_port=a.port,
                          inbrowser=False, **launch_kw)
