import csv
import re
from pathlib import Path

# 需要提取的实验列表（从 exp1 到 exp56）
exps = list(range(1, 57))

# 实验名称映射（根据 args.yaml 中的 model 字段推断）
def get_exp_name(exp):
    args_path = Path(f"runs/train/exp{exp}/args.yaml")
    if args_path.exists():
        with open(args_path, 'r') as f:
            content = f.read()
            # 查找 model 字段
            match = re.search(r"model:\s*(.+?)(?:\n|$)", content)
            if match:
                model = match.group(1).strip()
                # 简化名称
                if 'yolo11.yaml' in model:
                    return "YOLOv11基线"
                elif 'iRMB' in model and 'ASFYOLO' in model:
                    return "iRMB+ASFYolo"
                elif 'iRMB' in model:
                    return "+ iRMB"
                elif 'Biformer' in model:
                    return "Biformer"
                elif 'CGA' in model:
                    return "CGA"
                elif 'HSFPN' in model:
                    return "HSFPN"
                elif 'SlimNeck' in model:
                    return "SlimNeck"
                elif 'SDI' in model:
                    return "SDI"
                elif 'ASFYOLO' in model:
                    return "ASFYolo"
                elif 'EMA' in model:
                    return "EMA"
                elif 'DLKA' in model:
                    return "DLKA"
                elif 'DAT' in model:
                    return "DAT"
                elif 'CAA' in model:
                    return "CAA"
                elif 'FLA' in model:
                    return "FLA"
                elif 'iEMA' in model:
                    return "iEMA"
                elif 'ADNet' in model:
                    return "ADNet"
                elif 'AIFI' in model:
                    return "AIFI-DETR"
                elif 'CCFM' in model:
                    return "CCFM"
                elif 'Gold' in model:
                    return "Gold"
                elif 'RepGFPN' in model:
                    return "RepGFPN"
                elif 'BiFPN' in model:
                    return "BiFPN"
                elif 'MAB' in model:
                    return "MAB"
                else:
                    return model.split('/')[-1].replace('.yaml', '')
    return f"exp{exp}"

print("=" * 90)
print("所有实验 mAP50 最大值提取结果")
print("=" * 90)
print(f"{'实验':<8} {'方案':<35} {'mAP50最大值':<12} {'参数量(M)':<10} {'推理速度(ms)':<12}")
print("-" * 90)

results = []

for exp in exps:
    exp_path = Path(f"runs/train/exp{exp}/results.csv")
    
    if exp_path.exists():
        with open(exp_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            
            # 查找 mAP50 列
            map50_idx = None
            for i, col in enumerate(header):
                col_clean = col.strip()
                if col_clean == 'metrics/mAP50(B)':
                    map50_idx = i
                    break
            
            if map50_idx is not None:
                max_map50 = 0
                for row in reader:
                    try:
                        if len(row) > map50_idx:
                            val = float(row[map50_idx])
                            if val > max_map50:
                                max_map50 = val
                    except:
                        pass
                
                name = get_exp_name(exp)
                
                # 尝试从 args.yaml 读取 batch（作为参考）
                args_path = Path(f"runs/train/exp{exp}/args.yaml")
                batch = "?"
                if args_path.exists():
                    with open(args_path, 'r') as f:
                        content = f.read()
                        match = re.search(r"batch:\s*(\d+)", content)
                        if match:
                            batch = match.group(1)
                
                # 参数量和推理速度需要从验证结果图片中手动获取
                params = "?"
                speed = "?"
                
                # 对于已知的实验，填入已知数据
                if exp == 4:
                    params = "2.58"
                    speed = "0.9"
                elif exp == 16:
                    params = "2.60"
                    speed = "0.8"
                elif exp == 34:
                    params = "2.60"
                    speed = "1.1"
                
                print(f"exp{exp:<6} {name:<35} {max_map50*100:.2f}%{'':<6} {params:<10} {speed:<12}")
                results.append({
                    'exp': exp,
                    'name': name,
                    'mAP50_max': max_map50 * 100,
                    'params': params,
                    'speed': speed
                })
            else:
                print(f"exp{exp:<6} {'未找到mAP50列':<35} -")
    else:
        # 文件不存在，跳过
        pass

print("=" * 90)

# 按 mAP50 排序输出
print("\n按 mAP50 排序（从高到低，只显示有数据的实验）：")
print("-" * 60)
sorted_results = sorted([r for r in results if r['mAP50_max'] > 0], key=lambda x: x['mAP50_max'], reverse=True)
for i, r in enumerate(sorted_results, 1):
    print(f"{i:2}. exp{r['exp']:<5} {r['name']:<35} {r['mAP50_max']:.2f}%")