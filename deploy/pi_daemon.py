#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""24/7 background detector for the Raspberry Pi.

It owns the camera, runs NCNN continuously, and publishes:
  runtime/latest.jpg         latest annotated frame (atomic replace)
  runtime/status.json        timestamp / fps / counts / classes / camera flag
  runtime/detections-*.csv   one row per detected box (daily file)
  runtime/control.json       read from the web UI (source/conf/iou)

Run it with systemd (mulberry-daemon.service). The web UI only reads files,
so multiple browsers do not each start a separate inference loop.
"""
import csv
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

import pi_app  # noqa: E402

SHM = Path("/dev/shm/mulberry") if Path("/dev/shm").is_dir() else ROOT / "runtime"
LOG_DIR = ROOT / "runtime"
SHM.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
LATEST = SHM / "latest.jpg"
STATUS = SHM / "status.json"
CONTROL = SHM / "control.json"

TARGET_FPS = 2.0
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45
CAMERA_RETRY_SEC = 15.0
LOG_KEEP_DAYS = 7
LOG_EVERY_SEC = 5.0


def atomic_json(path, obj):
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)


def read_control():
    try:
        with open(CONTROL, encoding="utf-8") as f:
            c = json.load(f)
        source = c.get("source", pi_app.SRC_CAM)
        conf = float(c.get("conf", DEFAULT_CONF))
        iou = float(c.get("iou", DEFAULT_IOU))
        return source, max(0.01, min(0.99, conf)), max(0.01, min(0.99, iou))
    except Exception:
        return pi_app.SRC_CAM, DEFAULT_CONF, DEFAULT_IOU


def open_camera():
    idx = pi_app._find_camera()
    if idx is None:
        return None
    import cv2

    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    return cap


_DEMO_CACHE = {}


def demo_frame(state):
    files = pi_app.MON_IMAGES
    if not files:
        return None, "no-demo-image"
    path = files[state["i"] % len(files)]
    state["i"] += 1
    try:
        if path not in _DEMO_CACHE:
            im = Image.open(path).convert("RGB")
            im.thumbnail((640, 640), Image.BILINEAR)
            _DEMO_CACHE[path] = im
        return _DEMO_CACHE[path].copy(), "demo:" + os.path.basename(path)
    except Exception:
        return None, "demo-read-failed"


def cleanup_logs():
    cutoff = time.time() - LOG_KEEP_DAYS * 86400
    for p in LOG_DIR.glob("detections-*.csv"):
        try:
            if p.stat().st_mtime < cutoff:
                p.unlink()
        except Exception:
            pass


def append_log(rows, stamp, source):
    path = LOG_DIR / f"detections-{datetime.now():%Y%m%d}.csv"
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "source", "class_id", "class_cn", "class_en", "conf",
                        "x0", "y0", "x1", "y1"])
        for k, s, x0, y0, x1, y1 in rows:
            w.writerow([stamp, source, k, pi_app.CN[pi_app.NAMES[k]], pi_app.NAMES[k],
                        f"{s:.4f}", int(x0), int(y0), int(x1), int(y1)])


def main():
    print("mulberry daemon starting", flush=True)
    cleanup_logs()
    state = {"i": 0}
    cap = None
    last_cam_try = 0.0
    fps_n, fps_t0, fps = 0, time.time(), 0.0
    last_status = 0.0
    last_log = 0.0

    while True:
        t0 = time.time()
        source, conf, iou = read_control()

        if source == pi_app.SRC_DEMO:
            if cap is not None:
                cap.release()
                cap = None
        elif cap is None and t0 - last_cam_try >= CAMERA_RETRY_SEC:
            last_cam_try = t0
            cap = open_camera()
            if cap is None:
                print("camera not found; using demo images", flush=True)

        if cap is not None:
            ok, frame = cap.read()
            if not ok:
                print("camera read failed; release and retry", flush=True)
                cap.release()
                cap = None
                pil, source_tag = demo_frame(state)
                camera = False
            else:
                pil = Image.fromarray(frame[:, :, ::-1])  # BGR -> RGB
                source_tag = "usb-camera"
                camera = True
        else:
            pil, source_tag = demo_frame(state)
            camera = False

        if pil is None:
            time.sleep(1.0)
            continue

        try:
            boxes = pi_app.detect(pil, conf, iou)
        except Exception as e:
            print(f"detect failed: {e}", flush=True)
            traceback.print_exc()
            time.sleep(1.0)
            continue

        ann = pi_app.draw(pil, boxes)
        tmp = str(LATEST) + ".tmp.jpg"
        ann.save(tmp, quality=88, optimize=True)
        os.replace(tmp, LATEST)

        fps_n += 1
        dt = time.time() - fps_t0
        if dt >= 3.0:
            fps = fps_n / dt
            fps_n, fps_t0 = 0, time.time()

        now = time.time()
        classes = {}
        for k, s, *_ in boxes:
            cn = pi_app.CN[pi_app.NAMES[k]]
            classes[cn] = classes.get(cn, 0) + 1

        status = {
            "ts": now,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "fps": round(fps, 2),
            "count": len(boxes),
            "classes": classes,
            "camera": camera,
            "source": source_tag,
            "conf": conf,
            "iou": iou,
            "boxes": [[k, round(s, 3), int(x0), int(y0), int(x1), int(y1)]
                      for k, s, x0, y0, x1, y1 in boxes],
        }
        if now - last_status >= 0.5:
            atomic_json(STATUS, status)
            last_status = now

        if boxes and now - last_log >= LOG_EVERY_SEC:
            append_log(boxes, status["time"], source_tag)
            last_log = now

        elapsed = time.time() - t0
        time.sleep(max(0.0, 1.0 / TARGET_FPS - elapsed))


if __name__ == "__main__":
    main()
