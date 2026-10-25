# 数据集说明

本仓库 **不包含图像和标注文件**（原始目录约 7.2 GB，且涉及数据版权/隐私），只提供数据配置模板与统计信息。

## 目录结构

把数据放到仓库根目录下的 `datasets/mulberry/`：

```text
datasets/mulberry/
├── train/
│   ├── images/          # 1378 张
│   └── labels/          # YOLO 格式 .txt
└── valid/
    ├── images/          # 645 张
    └── labels/
```

`dataset/mulberry.yaml` 已按这个结构写好：

```yaml
path: ../datasets/mulberry
train: train/images
val: valid/images
```

如果你的数据放在别处，修改 `path` 或训练时传 `--data /your/path/data.yaml`。

## 类别

| ID | 英文名 | 中文名 |
| --- | --- | --- |
| 0 | Anthracnose | 炭疽病 |
| 1 | Healthy | 健康叶片 |
| 2 | Inchworm | 尺蠖 |
| 3 | Bacterial Wilt | 青枯病 |
| 4 | Yellow Melyrid Leaf Beetle | 黄斑叶甲 |
| 5 | Willow Blue Leaf Beetle | 柳蓝叶甲 |

## 统计（按当前 `dataset0224` 目录实际计数）

| Split | Images | Instances | Anthracnose | Healthy | Inchworm | Bacterial Wilt | Yellow Melyrid | Willow Blue |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 1378 | 1938 | 671 | 497 | 229 | 281 | 127 | 133 |
| valid | 645 | 1003 | 338 | 357 | 54 | 138 | 52 | 64 |
| **total** | **2023** | **2941** | **1009** | **854** | **283** | **419** | **179** | **197** |

说明：

- 训练/验证按约 7:3 划分，两个 split 的文件名无重叠。
- 论文表 1 的“2140 张 / 2962 个实例”与当前数据目录不一致；本仓库以实际文件计数为准，投稿前建议核对论文中的数据统计口径。
- 验证集各类别指标见 [`results/per_class.csv`](../results/per_class.csv)。

## 获取数据

数据不随仓库分发。需要复现实验或合作研究，请联系作者（见仓库主页）说明用途。
