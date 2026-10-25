#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Validate a trained checkpoint on the mulberry validation split.

Example:
    python val.py --weights weights/best.pt --data dataset/mulberry.yaml
"""
import argparse
import json
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent


def parse_args():
    ap = argparse.ArgumentParser(description="Validate the improved mulberry detector")
    ap.add_argument("--weights", default=str(ROOT / "weights/best.pt"))
    ap.add_argument("--data", default=str(ROOT / "dataset/mulberry.yaml"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0", help="0 / cpu")
    ap.add_argument("--project", default="runs/val")
    ap.add_argument("--name", default="exp")
    ap.add_argument("--exist-ok", action="store_true")
    ap.add_argument("--save-json", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    if not Path(args.weights).exists():
        raise FileNotFoundError(f"weights not found: {args.weights}")
    if not Path(args.data).exists():
        raise FileNotFoundError(f"dataset config not found: {args.data}")

    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data,
        split="val",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        save_json=args.save_json,
    )
    print(json.dumps(metrics.results_dict, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
