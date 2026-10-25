import os
from collections import defaultdict

# 修改为你的验证集 labels 路径
label_dir = r"datasets/mulberry/valid/labels"

class_counts = defaultdict(int)

# 类别名称映射（根据你的数据集顺序）
class_names = {
    0: "炭疽病 (Anthracnose)",
    1: "健康叶片 (Healthy)",
    2: "尺蠖 (Inchworm)",
    3: "青枯病 (Bacterial Wilt)",
    4: "黄斑叶甲 (Yellow Melyrid Leaf Beetle)",
    5: "柳蓝叶甲 (Willow Blue Leaf Beetle)"
}

for file in os.listdir(label_dir):
    if file.endswith('.txt'):
        with open(os.path.join(label_dir, file), 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_id = int(parts[0])
                    class_counts[class_id] += 1

print("验证集各类别实例数：")
print("-" * 50)
for class_id in sorted(class_counts.keys()):
    name = class_names.get(class_id, f"类别{class_id}")
    print(f"{name}: {class_counts[class_id]}")
print("-" * 50)
print(f"总计: {sum(class_counts.values())}")