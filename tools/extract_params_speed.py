from ultralytics import YOLO
import pandas as pd
import os

# 数据集配置文件路径（修改为你的实际路径）
DATA_YAML = r"dataset/mulberry.yaml"  # 或者 dataset/mulberry.yaml

# 需要验证的模型列表
# 格式：[(模型名称, best.pt 路径), ...]
models = [
    ("YOLOv11 基线", "runs/train/exp4/weights/best.pt"),
    ("YOLOv11-C2PSA-iRMB", "runs/train/exp16/weights/best.pt"),
    ("YOLOv11-C2PSA-iRMB-ASFYOLO", "runs/train/exp34/weights/best.pt"),
    # 👇 在这里添加你其他需要对比的模型
    ("Biformer", "runs/train/exp50/weights/best.pt"),
    ("HSFPN", "runs/train/exp19/weights/best.pt"),
    # ... 继续添加
]

results = []

for name, pt_path in models:
    if not os.path.exists(pt_path):
        print(f"⚠️ 文件不存在，跳过: {pt_path}")
        continue
    
    print(f"正在验证: {name}")
    model = YOLO(pt_path)
    val_results = model.val(data=DATA_YAML, verbose=True)
    
    results.append({
        "模型名称": name,
        "mAP50": round(val_results.box.map50, 4),
        "mAP50-95": round(val_results.box.map, 4),
        "精确率 P": round(val_results.box.mp, 4),
        "召回率 R": round(val_results.box.mr, 4),
        "参数量": model.model.model[-1].np,  # 近似
    })

# 打印汇总表格
print("\n" + "="*80)
print("验证结果汇总")
print("="*80)
df = pd.DataFrame(results)
print(df.to_string(index=False))

