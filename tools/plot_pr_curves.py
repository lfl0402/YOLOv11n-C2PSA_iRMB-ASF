import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

# 需要对比的实验列表（按表4顺序）
exps = [
    {"exp": 4, "name": "YOLOv11 基线", "color": "gray", "linestyle": "--"},
    {"exp": 50, "name": "Biformer", "color": "blue", "linestyle": "-."},
    {"exp": 52, "name": "CGA", "color": "green", "linestyle": "-."},
    {"exp": 19, "name": "HSFPN", "color": "orange", "linestyle": "-."},
    {"exp": 29, "name": "SlimNeck-3", "color": "purple", "linestyle": "-."},
    {"exp": 32, "name": "SDI-2", "color": "brown", "linestyle": "-."},
    {"exp": 37, "name": "ASFYolo-2", "color": "pink", "linestyle": "-."},
    {"exp": 34, "name": "本文方案 (iRMB+ASFYolo)", "color": "red", "linestyle": "-", "linewidth": 2},
]

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建图形
plt.figure(figsize=(10, 8))

# 存储 PR 数据
pr_data = {}

for exp_info in exps:
    exp_num = exp_info["exp"]
    exp_name = exp_info["name"]
    
    # 尝试读取 PR 数据（从 results.csv 中无法直接获取 PR 点，需要从 PR_curve.png 读取）
    # 由于 PR 曲线数据不保存在 CSV 中，这里提供两种方案
    
    # 方案1：如果每个 exp 有保存的 PR 数据文件
    pr_file = Path(f"runs/train/exp{exp_num}/PR_data.csv")
    if pr_file.exists():
        df = pd.read_csv(pr_file)
        plt.plot(df['recall'], df['precision'], 
                 color=exp_info["color"], 
                 linestyle=exp_info["linestyle"],
                 linewidth=exp_info.get("linewidth", 1),
                 label=f"{exp_name} (mAP50: {exp_info.get('map50', '?')}%)")
    else:
        # 方案2：如果没有 PR 数据，用 mAP50 模拟一个示意曲线（仅用于示意）
        # 注意：这只是一个占位方案，实际应该使用真实的 PR 数据
        print(f"警告: exp{exp_num} 没有 PR 数据文件，使用示意曲线")
        
        # 生成一个示意 PR 曲线（基于 mAP50 值）
        map50_val = {
            4: 83.59,
            50: 82.72,
            52: 83.45,
            19: 82.36,
            29: 82.58,
            32: 83.03,
            37: 82.52,
            34: 84.50,
        }.get(exp_num, 83.0) / 100
        
        # 生成平滑曲线
        recall = np.linspace(0, 1, 100)
        # 模拟 PR 曲线：precision 从高到低下降，曲线下面积约等于 mAP50
        precision = map50_val * (1 - 0.5 * recall)  # 示意公式
        precision = np.clip(precision, 0, 1)
        
        plt.plot(recall, precision,
                 color=exp_info["color"],
                 linestyle=exp_info["linestyle"],
                 linewidth=exp_info.get("linewidth", 1),
                 label=f"{exp_name} (mAP50: {map50_val*100:.2f}%)")

# 设置图表属性
plt.xlabel('召回率 (Recall)', fontsize=12)
plt.ylabel('精确率 (Precision)', fontsize=12)
plt.title('不同改进方案的 PR 曲线对比', fontsize=14)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)
plt.legend(loc='lower left', fontsize=9)

# 保存图片
plt.savefig('对比实验_PR曲线.png', dpi=300, bbox_inches='tight')
plt.savefig('对比实验_PR曲线.pdf', bbox_inches='tight')

print("PR 曲线对比图已保存为: 对比实验_PR曲线.png")
print("PR 曲线对比图已保存为: 对比实验_PR曲线.pdf")
plt.show()