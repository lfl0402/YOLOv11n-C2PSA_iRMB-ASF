#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""树莓派 SSH 助手(供部署流程使用)。

凭据从环境变量读取, 不写入任何文件:
    PI_HOST  默认 raspberrypi.local
    PI_USER  默认 pi
    PI_PASS  登录密码(必需)

用法:
    python pi_ssh.py "uname -a"              # 以普通用户执行
    python pi_ssh.py --sudo "apt update"     # 以 sudo 执行(密码自动喂给 sudo -S)
    python pi_ssh.py --timeout 3600 --sudo "make -j4"   # 长任务(秒)

注意: 长任务请用 --timeout 放大超时, 否则本地会先断开。
"""
import os
import sys
import time

import paramiko

HOST = os.environ.get("PI_HOST", "raspberrypi.local")
USER = os.environ.get("PI_USER", "pi")
PASS = os.environ.get("PI_PASS", "")
PORT = int(os.environ.get("PI_PORT", "22"))


def connect(retries=5, delay=3):
    """连接树莓派。.local 走 mDNS, 解析偶发失败, 因此带重试。"""
    last = None
    for i in range(retries):
        try:
            cli = paramiko.SSHClient()
            cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            cli.connect(HOST, port=PORT, username=USER, password=PASS,
                        timeout=30, banner_timeout=30, auth_timeout=30,
                        look_for_keys=False, allow_agent=False)
            return cli
        except Exception as e:  # 含 socket.gaierror(mDNS 解析失败)
            last = e
            if i < retries - 1:
                time.sleep(delay)
    raise last


def run(cmd, use_sudo=False, timeout=120):
    cli = connect()
    try:
        if use_sudo:
            cmd = f"echo '{PASS}' | sudo -S -p '' bash -lc {shell_quote(cmd)}"
        else:
            cmd = f"bash -lc {shell_quote(cmd)}"
        _, stdout, stderr = cli.exec_command(cmd, timeout=timeout, get_pty=False)
        out = stdout.read().decode("utf-8", "replace")
        err = stderr.read().decode("utf-8", "replace")
        rc = stdout.channel.recv_exit_status()
        sys.stdout.write(out)
        if err.strip():
            sys.stdout.write("\n--- stderr ---\n" + err)
        sys.stdout.write(f"\n--- exit={rc} ---\n")
        return rc
    finally:
        cli.close()


def shell_quote(s):
    return "'" + s.replace("'", "'\"'\"'") + "'"


if __name__ == "__main__":
    args = sys.argv[1:]
    use_sudo = False
    timeout = 120
    cmd = None
    i = 0
    while i < len(args):
        if args[i] == "--sudo":
            use_sudo = True
        elif args[i] == "--timeout":
            i += 1
            timeout = int(args[i])
        else:
            cmd = args[i]
        i += 1
    if not cmd:
        print(__doc__)
        sys.exit(2)
    if not PASS:
        print("需要设置环境变量 PI_PASS")
        sys.exit(2)
    sys.exit(run(cmd, use_sudo, timeout))
