#!/bin/bash

# Train with nano model for speed, 100 epochs, early stopping
yolo task=detect mode=train \
    model=yolov8n.pt \
    data=data/bangla_taka.yaml \
    epochs=100 \
    imgsz=640 \
    batch=16 \
    name=taka_banknotes_v1 \
    patience=20 \
    device=cpu  # Use CPU since no GPU is available

# Copy best model
mkdir -p weights
cp runs/detect/taka_banknotes_v1/weights/best.pt weights/bangla_taka_yolov8.pt