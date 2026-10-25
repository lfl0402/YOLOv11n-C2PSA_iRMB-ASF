# 致谢与第三方代码

本项目建立在以下开源工作之上。上传/再分发时请同时遵守各上游许可证。

## Ultralytics YOLO

- 项目：Ultralytics YOLO / YOLO11
- 地址：https://github.com/ultralytics/ultralytics
- 许可证：AGPL-3.0
- 说明：本仓库的 `ultralytics/` 是在其源码基础上修改得到的，主要新增 `nn/Addmodules`、`cfg/models/11/MY` 以及模型注册代码。按 AGPL-3.0，分发修改后的源码需保留许可证并公开源代码。

## iRMB

- 论文：Zhang J, Li X, Li J, et al. Rethinking Mobile Block for Efficient Attention-based Models. ICCV 2023.
- 说明：`ultralytics/nn/Addmodules/iRMB.py` 中的 iRMB / `C2PSA_iRMB` 参考该论文及公开实现改写。

## ASF-YOLO

- 论文：Kang M, Ting C M, Ting F F, et al. ASF-YOLO: A Novel YOLO Model with Attentional Scale Sequence Fusion for Cell Instance Segmentation. arXiv:2312.06458, 2023.
- 地址：https://arxiv.org/abs/2312.06458
- 说明：`ultralytics/nn/Addmodules/ASFYOLO.py` 中的 `Zoom_cat` / `ScalSeq` 参考 ASF-YOLO 的跨尺度融合思想实现；部署时为适配 NCNN 做了 2D 等价改写。

## YOLOv11 改进模块合集

- 来源：snu77 的 YOLOv11 改进项目/CSDN 专栏
- 说明：`ultralytics/nn/Addmodules/` 中除 iRMB/ASFYOLO 外的对比模块（EMA、CAA、DAT、DLKA、Gold-YOLO、RepGFPN 等）来自该合集，仅用于横向对比实验。

## 数据集

桑叶病虫害图像由作者收集/标注，不随仓库分发。若数据来自第三方，请在此补充来源与授权信息。

## 引用

如果本仓库对你的研究有帮助，请引用论文与 Ultralytics：

```bibtex
@article{liang2026mulberry,
  title   = {基于改进YOLO11n的桑叶病虫害检测方法研究},
  author  = {梁凤玲 and 韦日坤},
  journal = {物联网技术},
  year    = {2026},
  volume  = {16},
  number  = {19},
  doi     = {10.16667/j.issn.2095-1302.2026.19.008}
}

@misc{liang2026yolov11n,
  title  = {YOLOv11n-C2PSA-iRMB-ASF: Lightweight Mulberry Pest and Disease Detection with Edge Deployment},
  author = {Liang, Fengling and Wei, Rikun},
  year   = {2026},
  url    = {https://github.com/lfl0402/YOLOv11n-C2PSA_iRMB-ASF}
}

@software{ultralytics2024yolo11,
  title  = {Ultralytics YOLO11},
  author = {Jocher, Glenn and Qiu, Jing},
  year   = {2024},
  url    = {https://github.com/ultralytics/ultralytics},
  license = {AGPL-3.0}
}
```
