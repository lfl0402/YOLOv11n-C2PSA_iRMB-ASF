# -*- coding: utf-8 -*-
"""测试树莓派端实时监控功能(Gradio 界面构建 + 抓帧检测)。"""
import pi_app as P

print("[1] 构建界面…", flush=True)
demo = P.build_ui()
print("    OK")

print("[2] 演示模式抓帧…", flush=True)
ann, st = P.monitor_frame(P.SRC_DEMO, 0.25, 0.45, True)
print("    图尺寸:", ann.size if ann else None)
print("    状态:", st)
ann2, st2 = P.monitor_frame(P.SRC_DEMO, 0.25, 0.45, True)
print("    第二帧:", st2)

print("[3] 暂停态…", flush=True)
print("   ", P.monitor_frame(P.SRC_DEMO, 0.25, 0.45, False)[1])

print("[4] 摄像头探测…", flush=True)
idx = P._find_camera()
print("    找到摄像头序号:", idx if idx is not None else "无(将自动回退演示模式)")
ann3, st3 = P.monitor_frame(P.SRC_CAM, 0.25, 0.45, True)
print("    自动模式状态:", st3)

print("\n测试通过")
