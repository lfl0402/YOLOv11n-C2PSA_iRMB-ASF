#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""电脑端检测脚本(开箱即用)。加载论文模型对图片/文件夹做检测并保存标注图。

用法(在 F:\\YOLOv11-2 目录下运行, 用 base 环境):
    python detect_pc.py                                  # 默认检测验证集前 12 张(演示)
    python detect_pc.py --source 某张图.jpg               # 检测单张
    python detect_pc.py --source 某个文件夹 --conf 0.3     # 检测整个文件夹
    python detect_pc.py --source assets/samples --all   # 全部 645 张

参数:
    --model   模型权重, 默认 runs/train/exp34/weights/best.pt(论文模型, mAP50=84.5%)
    --out     结果保存目录, 默认 runs/detect/pc
    --conf    置信度阈值, 默认 0.25
    --all     源为文件夹时, 默认只处理前 12 张; 加 --all 处理全部
"""
import argparse
import glob
import os
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # 本机 OpenMP 重复链接的已知问题

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG")


def main():
    ap = argparse.ArgumentParser(description="桑叶病虫害检测(电脑端)")
    ap.add_argument("--model", default="weights/best.pt")
    ap.add_argument("--source", default="assets/samples")
    ap.add_argument("--out", default=r"runs/detect/pc")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--all", action="store_true", help="文件夹源: 处理全部图片")
    a = ap.parse_args()

    if not os.path.exists(a.model):
        print(f"[错误] 找不到模型权重: {a.model}")
        print("       当前目录:", os.getcwd())
        return 1
    if not os.path.exists(a.source):
        print(f"[错误] 找不到检测源: {a.source}")
        return 1

    # 文件夹源默认只取前 12 张, 便于快速演示
    source = a.source
    n_total = None
    if os.path.isdir(a.source):
        files = sorted(f for f in glob.glob(os.path.join(a.source, "*")) if f.endswith(IMG_EXT))
        n_total = len(files)
        if not a.all:
            files = files[:12]
        if not files:
            print(f"[错误] 目录里没有图片: {a.source}")
            return 1
        source = files

    from ultralytics import YOLO
    model = YOLO(a.model)
    print(f"模型: {a.model}")
    print(f"检测: {a.source}" + (f"  (共 {n_total} 张, 本次处理 {len(source)} 张)" if n_total else ""))
    print("-" * 60)

    results = model.predict(source=source, imgsz=a.imgsz, conf=a.conf,
                            project=os.path.dirname(a.out), name=os.path.basename(a.out),
                            save=True, exist_ok=True, verbose=False)

    print("-" * 60)
    total_obj = 0
    for r in results:
        names = r.names
        cnt = {}
        for c in (r.boxes.cls.tolist() if r.boxes is not None else []):
            cnt[names[int(c)]] = cnt.get(names[int(c)], 0) + 1
        total_obj += sum(cnt.values())
        tag = ", ".join(f"{k}×{v}" for k, v in cnt.items()) or "无目标"
        print(f"  {os.path.basename(r.path):32s} {tag}")
    print("-" * 60)
    print(f"共处理 {len(results)} 张, 检出目标 {total_obj} 个")
    print(f"标注图已保存到: {os.path.abspath(a.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
