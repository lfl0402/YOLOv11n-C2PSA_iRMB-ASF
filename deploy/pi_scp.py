#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SFTP 传输助手(供部署流程使用)。凭据同 pi_ssh.py, 从环境变量读取。

用法:
    python pi_scp.py <本地路径1> [本地路径2 ...] <远程目录>     # 上传(目录递归)
    python pi_scp.py --get <远程路径> <本地目录>               # 下载(目录递归)
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
    """.local 走 mDNS, 解析偶发失败, 因此带重试。"""
    last = None
    for i in range(retries):
        try:
            cli = paramiko.SSHClient()
            cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            cli.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30,
                        look_for_keys=False, allow_agent=False)
            return cli
        except Exception as e:
            last = e
            if i < retries - 1:
                time.sleep(delay)
    raise last


def ensure_dir(sftp, path):
    parts = path.strip("/").split("/")
    cur = ""
    for p in parts:
        cur = cur + "/" + p
        try:
            sftp.stat(cur)
        except IOError:
            sftp.mkdir(cur)


def put(sftp, local, remote):
    if os.path.isdir(local):
        ensure_dir(sftp, remote)
        for name in sorted(os.listdir(local)):
            put(sftp, os.path.join(local, name), remote.rstrip("/") + "/" + name)
    else:
        sftp.put(local, remote)
        size = os.path.getsize(local)
        print(f"  {local} -> {remote}  ({size/1024:.0f} KB)", flush=True)


def get(sftp, remote, local):
    """递归下载。"""
    try:
        st = sftp.stat(remote)
    except IOError:
        print(f"远程路径不存在: {remote}")
        return
    if st.st_mode & 0o40000:  # 目录
        os.makedirs(local, exist_ok=True)
        for name in sorted(sftp.listdir(remote)):
            get(sftp, remote.rstrip("/") + "/" + name, os.path.join(local, name))
    else:
        os.makedirs(os.path.dirname(local) or ".", exist_ok=True)
        sftp.get(remote, local)
        print(f"  {remote} -> {local}  ({os.path.getsize(local)/1024:.0f} KB)", flush=True)


def main():
    args = sys.argv[1:]
    if args and args[0] == "--get":
        if len(args) != 3:
            print(__doc__)
            return 2
        cli = connect()
        try:
            sftp = cli.open_sftp()
            get(sftp, args[1], args[2])
            sftp.close()
            print("下载完成")
            return 0
        finally:
            cli.close()

    if len(args) < 2:
        print(__doc__)
        return 2
    remote_dir = args[-1]
    locals_ = args[:-1]

    cli = connect()
    try:
        sftp = cli.open_sftp()
        ensure_dir(sftp, remote_dir)
        for l in locals_:
            if not os.path.exists(l):
                print(f"本地路径不存在: {l}")
                return 1
            put(sftp, l, remote_dir.rstrip("/") + "/" + os.path.basename(l))
        sftp.close()
        print("上传完成")
        return 0
    finally:
        cli.close()


if __name__ == "__main__":
    sys.exit(main())
