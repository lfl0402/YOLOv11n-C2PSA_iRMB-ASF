# 树莓派 4B 部署（NCNN + Web）

本目录是边缘端部署的完整代码与已转换模型。树莓派只做 NCNN 推理，不安装 PyTorch。

## 实测

| 项目 | 值 |
| --- | --- |
| 设备 | Raspberry Pi 4B（4 GB，Cortex-A72 四核） |
| 输入 | 640×640，FP32 |
| 推理引擎 | NCNN（源码编译，与 pnnx 版本配套） |
| 单帧耗时 | **438 ms/帧（约 2.3 FPS）** |
| 结果一致性 | 与 PC 端 PyTorch 检测结果逐项一致 |
| Web 端口 | 7860 |

> Pi 4B 的 Cortex-A72 是 ARMv8.0，没有 FP16/INT8 点积指令，因此使用 FP32；不要给 NCNN 加 `NCNN_ARM82` 或 `-mcpu=cortex-a76`，否则会 `SIGILL`。

## 文件

```text
deploy/
├── pi_app.py                 # Gradio 网页（只读后台最新帧 + 单张/批量检测）
├── pi_daemon.py              # 24/7 后台采集/推理 daemon
├── systemd/                  # mulberry.service + mulberry-daemon.service
├── webapp.sh                 # 手动 start/stop/restart/status/log
├── requirements-pi.txt
├── model/                    # NCNN param/bin + TorchScript（已转换）
├── images/                   # 无摄像头时的演示图（47/48/49.jpg）
├── out_pi/                   # 树莓派实测输出示例
├── export.py                 # PC 端: best.pt -> TorchScript -> pnnx -> NCNN
├── verify_ncnn.py            # PyTorch vs NCNN 数值一致性验证
├── validate_param.py         # 检查 param 中是否有 NCNN 不支持的算子
├── reference.py              # 纯 Python NCNN 参考实现
├── pi_ssh.py / pi_scp.py     # SSH/SFTP 小工具（读环境变量）
└── cpp/                      # C++ NCNN 推理与摄像头演示
```

## 1. 拷贝到树莓派

在 PC 仓库根目录：

```bash
ssh pi@raspberrypi.local "mkdir -p ~/mulberry"
scp -r deploy/* pi@raspberrypi.local:~/mulberry/
```

或者用 `deploy/pi_scp.py`（先设置环境变量）：

```bash
set PI_HOST=raspberrypi.local
set PI_USER=pi
set PI_PASS=你的密码
python deploy/pi_scp.py deploy pi@raspberrypi.local:~/mulberry
```

> `pi_ssh.py` / `pi_scp.py` 从 `PI_HOST`、`PI_USER`、`PI_PASS` 读取连接信息，不保存密码。

## 2. 安装依赖

```bash
ssh pi@raspberrypi.local
cd ~/mulberry
sudo apt update
sudo apt install -y python3-venv python3-pip
python3 -m venv venv
./venv/bin/pip install -U pip
./venv/bin/pip install -r requirements-pi.txt
```

如果 `ncnn` 的 pip 轮子在你的系统上不可用，按 NCNN 官方文档源码编译 Python 绑定；版本要与生成模型的 pnnx 版本匹配。

## 3. 启动（推荐 systemd，24/7）

`deploy/systemd/` 里有两个服务：

| 服务 | 作用 |
| --- | --- |
| `mulberry.service` | Gradio 网页，端口 7860 |
| `mulberry-daemon.service` | 后台 24/7 检测（摄像头 → NCNN → runtime/） |

安装并开机自启：

```bash
sudo cp deploy/systemd/*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now mulberry mulberry-daemon
sudo systemctl status mulberry mulberry-daemon --no-pager
```

> 服务里的 `User=lfl`、`/home/lfl/mulberry` 是按本机写的；用户名/路径不同时先改 `deploy/systemd/*.service`。

访问：

- 本机：`http://127.0.0.1:7860`
- 局域网电脑/手机：`http://<树莓派IP>:7860`
- mDNS：`http://raspberrypi.local:7860`

手机浏览器“添加到主屏幕”即可像 App 一样使用。

### 24/7 后台检测

`pi_daemon.py` 常驻运行，自己打开摄像头并推理；网页只读取最新帧，所以多个浏览器不会各跑一遍模型。

运行时文件：

| 文件 | 内容 |
| --- | --- |
| `/dev/shm/mulberry/latest.jpg` | 最新标注图（内存盘，避免写 SD 卡） |
| `/dev/shm/mulberry/status.json` | 时间、FPS、目标数、类别 |
| `/dev/shm/mulberry/control.json` | 网页上的视频源/置信度/IoU |
| `~/mulberry/runtime/detections-YYYYMMDD.csv` | 每个检测框一行，最多每 5 秒写一次，保留 7 天 |

> `latest/status/control` 放在 `/dev/shm`，重启后丢失但 daemon 会重建；CSV 日志在磁盘。

摄像头：

- USB 摄像头接到 Pi，出现 `/dev/video0` 后 15 秒内自动接管；
- 没有摄像头时循环播放 `images/` 演示图，状态里会显示“演示模式”；
- CSI 排线摄像头（Pi Camera）当前代码不直接支持，需要 `libcamera`/`picamera2` 适配。

查看状态/日志：

```bash
journalctl -u mulberry-daemon -f
cat ~/mulberry/runtime/status.json
ls -l ~/mulberry/runtime/
```

### 手动启动（调试）

```bash
chmod +x webapp.sh
./webapp.sh start                  # 只启动网页
./venv/bin/python pi_daemon.py     # 另开终端启动后台检测
```

## 4. 重新导出模型（PC 端）

`best.pt` → TorchScript → pnnx → NCNN：

```bash
python deploy/export.py --weights weights/best.pt
```

依赖：

- `pnnx` 可执行文件在 `PATH` 中（https://github.com/pnnx/pnnx/releases）
- `ncnnoptimize`（可选，用于进一步优化）
- Python 端 `torch`、`opencv-python`、`numpy`

导出脚本会自动：

1. 把 Neck 中 NCNN 不支持的 5 维 `ScalSeq` 路径替换为数学等价的 2D 实现；
2. 校验 eager / traced 输出一致性；
3. 生成 `deploy/model/yolov11_mulberry.torchscript.pt` 与 `.ncnn.param/.bin`；
4. 用真实图片做解码自检。

数值一致性验证：

```bash
python deploy/verify_ncnn.py --mode torch
python deploy/verify_ncnn.py --mode ncnn
python deploy/verify_ncnn.py --mode compare
```

## 5. C++ 版本（可选）

需要 OpenCV 与 NCNN C++ 库：

```bash
cd cpp
mkdir -p build && cd build
cmake ..
make -j4
./mulberry_bench ../model/yolov11_mulberry.torchscript.ncnn.param \
                  ../model/yolov11_mulberry.torchscript.ncnn.bin
./mulberry_detect ../model/yolov11_mulberry.torchscript.ncnn.param \
                  ../model/yolov11_mulberry.torchscript.ncnn.bin \
                  ../images --out ../out_cpp
```

摄像头实时演示：

```bash
./mulberry_detect ../model/yolov11_mulberry.torchscript.ncnn.param \
                  ../model/yolov11_mulberry.torchscript.ncnn.bin \
                  --cam 0
```

## 常见问题

- **`ModuleNotFoundError: ncnn`**：树莓派 venv 里没有装 `ncnn`，或 pip 轮子与系统架构不匹配。用 `./venv/bin/pip install ncnn` 重装。
- **`load_param` 失败**：`param/bin` 必须来自同一次 pnnx 转换，不能混用不同版本。
- **页面能开但推理报错**：先跑 `./venv/bin/python pi_app.py --selftest images/47.jpg`，看是否是模型/预处理问题。
- **摄像头打不开**：确认 `/dev/video0` 存在、用户有 `video` 组权限；没有摄像头时用演示模式即可。
- **PC 端与 Pi 端结果不一致**：检查输入尺寸、LetterBox 灰边值（114）、BGR/RGB 顺序和 NMS 阈值是否一致；`verify_ncnn.py` 可逐项对比。
