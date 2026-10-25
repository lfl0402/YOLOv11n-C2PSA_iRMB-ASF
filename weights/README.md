# 权重说明

`best.pt` 是论文最终模型（`exp34`）的最佳检查点：

| 项目 | 值 |
| --- | --- |
| 模型 | YOLOv11n + C2PSA-iRMB + ASF |
| 配置 | `ultralytics/cfg/models/11/MY/yolo11-C2PSA-iRMB-ASFYOLO-1.yaml` |
| 最佳轮次 | 226 / 276（早停） |
| 验证集 | 645 张，1003 个实例，6 类 |
| Precision | 86.82% |
| Recall | 79.16% |
| mAP50 | 84.50% |
| mAP50-95 | 57.43% |
| 参数量 | 2.98 M（实测） |
| GFLOPs | 7.83（640×640） |
| SHA256 | `BAE6895A82BC7A9C06110E347AB821188F5F662607DCD626888454F4F4403545` |

复现验证：

```bash
python val.py --weights weights/best.pt --data dataset/mulberry.yaml --device 0
```

树莓派端使用的 NCNN 模型在 [`deploy/model/`](../deploy/model)：

- `yolov11_mulberry.torchscript.ncnn.param`
- `yolov11_mulberry.torchscript.ncnn.bin`

重新导出：

```bash
python deploy/export.py --weights weights/best.pt
```

权重以 AGPL-3.0 随本仓库分发；使用前请阅读根目录 [`LICENSE`](../LICENSE) 与 [`ACKNOWLEDGEMENTS.md`](../ACKNOWLEDGEMENTS.md)。
