# Ultralytics YOLO 🚀, AGPL-3.0 license
#
# 自定义模块集中导入。原版是 100 多行 `from .X import *`，其中不少模块依赖
# mmcv / basicsr / torchvision 等可选包，只要有一个导入失败，整个 ultralytics
# 包就导入不了。而本仓库训练出的 best.pt 里的类路径是
# `ultralytics.nn.Addmodules.xxx`，torch.load 反序列化时必须导入本包 ——
# 于是会抛 ModuleNotFoundError: mmcv，Ultralytics 又会误判“模型需要 mmcv”
# 去自动 pip install mmcv（在 Python 3.13 上必然编译失败）。
#
# 这里改成逐个模块容错导入：能用的照常注册（parse_model / torch.load 都能找到），
# 缺依赖的安静跳过。导入失败的模块名与原因记录在 _import_errors 里。

import importlib as _importlib

_MODULES = [
    "ACmix",
    "ADNet",
    "ADown",
    "AirNet",
    "AKConv",
    "AodPONONet",
    "Biformer",
    "BiFPN",
    "CARAFE",
    "CGAttention",
    "ContextGuided",
    "ConvNeXtV2",
    "CPAEnhancer",
    "CSWinTransformer",
    "DamoYOLO",
    "DAT",
    "DeformableLKA",
    "DiverseBranchBlock",
    "DSConv",
    "DualConv",
    "DWRSeg",
    "DynamicConv",
    "DynamicConvModule",
    "Dysample",
    "EfficientNetV1",
    "EfficientNetV2",
    "EMA",
    "EMO",
    "FasterBlock",
    "FasterNet",
    "FFANet",
    "FocalModulation",
    "FocusedLinearAttention",
    "GELAN",
    "GhostNetV1",
    "GhostNetV2",
    "Haar",
    "HetConv",
    "IAT",
    "iRMB",
    "KANConv2Dlayers",
    "LAE",
    "LightHGNetV2",
    "LSKA",
    "LSKNet",
    "MBTaylorFormer",
    "MLCA",
    "MLLA",
    "MLLABlock",
    "mobileNetV1",
    "mobileNetV2",
    "MoileNetV1",
    "MoileNetV2",
    "MoileNetV3",
    "MoileNetV4",
    "MSBYOLO",
    "MSDA",
    "ODConv",
    "OREPA",
    "PConv",
    "PEYOLO",
    "RCSOSA",
    "RepELAN",
    "RepViT",
    "Retinexformer",
    "RevColV1",
    "RFAConv",
    "RIDNet",
    "RTDETRv2Att",
    "SAConv",
    "ScConv",
    "SCINet",
    "SCSA",
    "SEAM",
    "SENetV1",
    "SENetV2",
    "ShuffleNetV1",
    "ShuffleNetV2",
    "SlimNeck",
    "SPDConv",
    "SwinTransformer",
    "TransNeXt",
    "TripleAttention",
    "UnfogNet",
    "UniRepLknet",
    "VanillaNet",
    "WTConv",
    "CBAM",
    "ECA",
    "CA",
    "GAM",
    "SDI",
    "ASFFHead",
    "AFPNHead",
    "AFPN4Head",
    "DynamicHead",
    "DynamicDCNv3Head",
    "DSConvHead",
    "FASFFHead",
    "EUCB",
    "MSCB",
    "MAB",
    "CAA",
    "iEMA",
    "GoldYOLO",
    "ASFYOLO",
    "HSFPN",
    "FreqFusion",
    "iAFF",
    "UIB",
]

_import_errors = {}
_loaded = []


def _star_import(name):
    """等价于 `from .<name> import *`，但失败时只记录、不抛出。"""
    try:
        module = _importlib.import_module(f".{name}", __package__)
    except Exception as e:  # 缺依赖、版本不兼容等
        _import_errors[name] = f"{type(e).__name__}: {e}"
        return
    names = getattr(module, "__all__", None) or [k for k in vars(module) if not k.startswith("_")]
    for k in names:
        globals()[k] = getattr(module, k)
    _loaded.append(name)


for _name in _MODULES:
    _star_import(_name)

if _import_errors:
    print(
        f"Addmodules: 已导入 {len(_loaded)}/{len(_MODULES)} 个自定义模块，"
        f"跳过 {len(_import_errors)} 个（缺少 mmcv / basicsr 等可选依赖，"
        f"不影响已导入模块，明细见 ultralytics.nn.Addmodules._import_errors）",
        flush=True,
    )

del _name
