# 实验结果与图表

本目录只保留论文相关的精选结果，不包含 61 次训练的完整 `runs/`（约 4 GB）。

## 文件

| 文件 | 内容 |
| --- | --- |
| `ablation.csv` | 论文表 2 的消融实验（基线 / +C2PSA-iRMB / +C2PSA-iRMB+ASF） |
| `per_class.csv` | 论文表 3 的验证集各类别指标 |
| `supplementary_comparison.csv` | 其他改进模块的横向对比（补充材料） |
| `all_runs.csv` | 全部有效训练 run 的最佳轮次指标 |
| `dataset_distribution.csv` | 数据集划分与类别实例数 |
| `comparison_PR_curve.png/.pdf` | 对比实验 PR 曲线 |
| `exp34/` | 最终模型的训练曲线、PR/F1/P/R 曲线、混淆矩阵、验证集预测图 |
| `exp4/`、`exp16/` | 基线、+C2PSA-iRMB 的 PR 曲线与训练曲线 |

## 指标口径

- 表中指标取 **best.pt 对应轮次**（`results.csv` 中 mAP50 最大的那一行），不是最后一轮。
- 最终模型：`exp34`，最佳轮次 226，早停于 276。
- 参数量/GFLOPs 使用 `ultralytics` 的 `model.info()` 在 640×640 输入下实测。
- 原始项目根目录的 `all_experiments_metrics.csv` 把 `mAP50-95` 列误当成了 `mAP50`（因此 exp34 显示为 57.43）；本目录的 CSV 已按 `results.csv` 重新计算并修正。

## 复现

```bash
python val.py --weights weights/best.pt --data dataset/mulberry.yaml --device 0
```

训练日志与完整 `results.csv` 在 `results/exp34/`；其余 run 的原始输出因体积原因未上传。
