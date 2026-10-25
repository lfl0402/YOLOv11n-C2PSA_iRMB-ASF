# -*- coding: utf-8 -*-
"""单张图片检测。

用法(在 F:\\YOLOv11-2 目录下运行):
    python run_test.py                       # 默认检测验证集里的 47.jpg
    python run_test.py 某张图.jpg             # 检测指定图片
    python run_test.py --show                # 检测后弹窗显示(需要桌面环境)
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
# 本机 torch 与 timm 重复链接 OpenMP 运行库, 不设这个变量会报 OMP: Error #15
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from ultralytics import YOLO

MODEL = "weights/best.pt"                 # 论文模型(mAP50=84.5%)
IMG = "assets/samples/47.jpg"                     # 默认检测的图片
OUT = "runs/detect/single/47.jpg"                           # 保存路径


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    show = "--show" in sys.argv

    img_path = args[0] if args else IMG
    if not os.path.exists(MODEL):
        print(f"[错误] 找不到模型: {os.path.abspath(MODEL)}")
        print("       请在 F:\\YOLOv11-2 目录下运行本脚本")
        return 1
    if not os.path.exists(img_path):
        print(f"[错误] 找不到图片: {os.path.abspath(img_path)}")
        return 1

    model = YOLO(MODEL)
    results = model(img_path, imgsz=640, conf=0.25, verbose=False)
    r = results[0]

    print(f"图片: {r.path}")
    print(f"检出 {len(r.boxes)} 个目标:")
    for i in range(len(r.boxes)):
        cls = r.names[int(r.boxes.cls[i])]
        conf = float(r.boxes.conf[i])
        x0, y0, x1, y1 = (int(v) for v in r.boxes.xyxy[i])
        print(f"  {cls:28s} conf={conf:.3f}  框=({x0},{y0},{x1},{y1})")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    r.save(OUT)
    print(f"\n标注图已保存: {os.path.abspath(OUT)}")

    if show:
        r.show()          # 弹窗显示; 默认关闭, 以免在没有桌面的环境下卡住
    return 0


if __name__ == "__main__":
    sys.exit(main())
