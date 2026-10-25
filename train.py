#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train YOLOv11n-C2PSA-iRMB-ASF on the mulberry pest dataset.

The default settings follow the paper: 640x640, 300 epochs, batch 48, SGD,
AMP enabled, close_mosaic=0, patience=50.

Example:
    python train.py --data dataset/mulberry.yaml --device 0
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "ultralytics/cfg/models/11/MY/yolo11-C2PSA-iRMB-ASFYOLO-1.yaml"
DEFAULT_DATA = ROOT / "dataset/mulberry.yaml"


def parse_args():
    ap = argparse.ArgumentParser(description="Train the improved mulberry detector")
    ap.add_argument("--model", default=str(DEFAULT_MODEL), help="model YAML or checkpoint")
    ap.add_argument("--data", default=str(DEFAULT_DATA), help="dataset YAML")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0", help="0 / 0,1 / cpu")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", action="store_true", help="cache images in RAM (needs enough memory)")
    ap.add_argument("--project", default="runs/train")
    ap.add_argument("--name", default="exp")
    ap.add_argument("--exist-ok", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model)
    data_path = Path(args.data)
    if not model_path.exists():
        raise FileNotFoundError(f"model config not found: {model_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"dataset config not found: {data_path}")

    model = YOLO(str(model_path))
    model.train(
        data=str(data_path),
        task="detect",
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        single_cls=False,
        close_mosaic=0,
        workers=args.workers,
        device=args.device,
        optimizer="SGD",
        amp=True,
        pretrained=True,
        patience=50,
        cache=args.cache,
        seed=args.seed,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
    )


if __name__ == "__main__":
    main()
