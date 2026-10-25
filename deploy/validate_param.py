# -*- coding: utf-8 -*-
"""校验 pnnx 生成的 ncnn param 是否可部署: 检查超 4 维算子、层类型清单、输入输出名。"""
import collections
import os

param = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model",
                     "yolov11_mulberry.torchscript.ncnn.param")
lines = [ln.rstrip("\n") for ln in open(param, encoding="utf-8") if ln.strip()]

magic, layer_count, blob_count = lines[0], lines[1], lines[2]
print(f"magic={magic}  layer_count/blob_count={layer_count}")

types = collections.Counter()
problems = []
inputs, outputs = [], []

for i, ln in enumerate(lines[3:], start=3):
    parts = ln.split()
    ltype = parts[0]
    types[ltype] += 1
    if ltype == "Input":
        inputs.extend(parts[3:])
    # 检查 Reshape 的 shape 参数(键 6= 或 0=..5=)维数
    if ltype == "Reshape":
        for kv in parts[4:]:
            if kv.startswith("6="):
                ndim = len(kv[2:].strip('"').split(","))
                if ndim > 4:
                    problems.append((i + 1, ln.strip()))
            elif kv.startswith("0="):
                # 旧式逐维写法: 0=..5= 最多 6 维, 统计出现了多少维
                dims = [p for p in parts[4:] if p.split("=")[0] in "012345"]
                if len(dims) > 4:
                    problems.append((i + 1, ln.strip()))
    if ltype == "Permute":
        for kv in parts[4:]:
            if kv.startswith("0="):
                order = kv[2:]
                if len(order) > 4:
                    problems.append((i + 1, ln.strip()))

print("\n层类型清单:")
for t, n in types.most_common():
    print(f"  {t:24s} x{n}")

print("\n输入 blob:", inputs)
print("疑似超 4 维算子:", len(problems))
for lineno, ln in problems:
    print(f"  行 {lineno}: {ln}")

# 最后一行通常是输出名
tail = lines[-1].split()
print("\n末层:", tail[0], "输出:", tail[3:])

print("\n判定:", "PASS —— 无超 4 维算子" if not problems else "FAIL —— 存在超 4 维算子")
