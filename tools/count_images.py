import os
from collections import defaultdict

# 修改为你的验证集 labels 路径
label_dir = r"datasets/mulberry/valid/labels"

# 类别名称映射
class_names = {
    0: "炭疽病 (Anthracnose)",
    1: "健康叶片 (Healthy)",
    2: "尺蠖 (Inchworm)",
    3: "青枯病 (Bacterial Wilt)",
    4: "黄斑叶甲 (Yellow Melyrid Leaf Beetle)",
    5: "柳蓝叶甲 (Willow Blue Leaf Beetle)"
}

# 统计每个类别出现在多少张图片中
class_image_counts = defaultdict(set)

for file in os.listdir(label_dir):
    if file.endswith('.txt'):
        with open(os.path.join(label_dir, file), 'r') as f:
            classes_in_this_image = set()
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_id = int(parts[0])
                    classes_in_this_image.add(class_id)
            for class_id in classes_in_this_image:
                class_image_counts[class_id].add(file)

print("验证集各类别图片数：")
print("-" * 50)
total_images = 0
for class_id in sorted(class_image_counts.keys()):
    count = len(class_image_counts[class_id])
    name = class_names.get(class_id, f"类别{class_id}")
    print(f"{name}: {count}")
    total_images += count
print("-" * 50)
print(f"注意：上述合计可能大于645，因为一张图可能包含多个类别")
print(f"实际验证集总图片数: {len(os.listdir(label_dir))}")