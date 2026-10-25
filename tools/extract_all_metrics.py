# find_and_eval.py
import os
import glob
from ultralytics import YOLO

DATA_YAML = "dataset/mulberry.yaml"

# 搜索所有可能的权重文件
weight_patterns = [
    "runs/train/*/weights/best.pt",
    "runs/train/exp*/weights/best.pt",
]

for pattern in weight_patterns:
    for weight_path in glob.glob(pattern):
        exp_name = weight_path.split("\\")[-3] if "\\" in weight_path else weight_path.split("/")[-3]
        print(f"\n评估: {exp_name}")
        
        model = YOLO(weight_path)
        results = model.val(data=DATA_YAML, device="0")
        
        print(f"  mAP50: {results.box.map50*100:.2f}%")
        print(f"  精确率: {results.box.p[0]*100:.2f}%")
        print(f"  召回率: {results.box.r[0]*100:.2f}%")