import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ========== 配置 ==========
MODEL_PATH = r"runs/train/exp34/weights/best.pt"
IMAGE_DIR = r"datasets/mulberry/valid/images"
SAVE_DIR = r"results_heatmap"
# =========================

# 创建保存文件夹
Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

# 加载模型
model = YOLO(MODEL_PATH)

# 处理每张图片
for img_path in Path(IMAGE_DIR).glob("*.jpg"):
    print(f"处理: {img_path.name}")
    
    # 检测并保存（带检测框的结果图）
    results = model(img_path, save=True, project=SAVE_DIR, name="detect", conf=0.25)
    
    print(f"  已保存: {SAVE_DIR}/detect/{img_path.name}")

print(f"\n完成！热力图保存在: {SAVE_DIR}")