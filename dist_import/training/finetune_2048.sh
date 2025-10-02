#!/usr/bin/env bash
set -euo pipefail

yolo segment train \
  model=runs_seg/tools_1024/weights/best.pt \
  data=dataset/data.yaml imgsz=2048 epochs=25 batch=4 device=0 \
  close_mosaic=10 degrees=10 \
  project=runs_seg name=tools_ft2048


