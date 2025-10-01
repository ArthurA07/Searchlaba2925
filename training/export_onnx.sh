#!/usr/bin/env bash
set -euo pipefail

yolo export model=runs_seg/tools_ft2048/weights/best.pt format=onnx
mkdir -p backend/models
mv runs_seg/tools_ft2048/weights/best.onnx backend/models/best.onnx
echo "[ok] ONNX exported to backend/models/best.onnx"


