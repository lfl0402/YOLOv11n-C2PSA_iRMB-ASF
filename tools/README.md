# 工具脚本

这些脚本用于统计、绘图和结果提取，不是训练/部署的必需项。部分脚本里的路径是按原项目 `runs/train/exp*` 写的，运行前请按你的目录调整。

| 脚本 | 用途 |
| --- | --- |
| `extract_results.py` | 从各 `runs/train/exp*/results.csv` 提取最佳轮次指标 |
| `plot_pr_curves.py` | 绘制多实验 PR 曲线 |
| `count_labels.py` | 统计标签文件与实例数 |
| `count_images.py` | 统计各 split 图像数 |
| `extract_params_speed.py` | 提取参数量/推理速度 |
| `extract_all_metrics.py` | 汇总所有 run 的指标 |
| `make_heatmap.py` | 生成检测热力图 |
| `fps.py` | 推理 FPS 测试 |
| `draw_loss.py` | 绘制损失曲线 |

运行示例：

```bash
python tools/extract_results.py
python tools/count_labels.py
```
