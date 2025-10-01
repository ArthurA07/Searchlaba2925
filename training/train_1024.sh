#!/usr/bin/env bash
set -euo pipefail

yolo segment train \
  model=yolov8s-seg.pt data=dataset/data.yaml \
  imgsz=1024 epochs=120 batch=16 device=0 \
  close_mosaic=10 hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 degrees=15 \
  project=runs_seg name=tools_1024


