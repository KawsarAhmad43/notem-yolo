# Notem: Bangladeshi Taka Banknote Detection Project

## Overview
This project detects and classifies Bangladeshi Taka banknotes (1, 2, 5, 10, 20, 50, 100, 500, 1000) using YOLOv8 for object detection and FastAPI for serving predictions. It supports single or multi-note images, with dataset prep, training, and API deployment.

## Features
- Dataset conversion to YOLO format with dummy/full-image labels (improve with manual annotation).
- YOLOv8 training on CPU/GPU.
- FastAPI API for image upload and detection (class, confidence, bbox, total value).
- Docker support.

## Project Structure
```bash
notem/
├── data/                # Data folders
│   ├── training/        # Class folders (1/, etc.) + takas.png
│   ├── testing/         # Test images
│   └── yolo_dataset/    # YOLO-ready (images/labels: train/val/test)
├── runs/                # YOLO outputs
├── weights/             # Trained models
├── src/                 # Scripts
│   ├── init.py
│   ├── dataset.py
│   └── utils.py
├── notebooks/           # Notebooks
│   └── 01_data_prep.ipynb
├── app/                 # FastAPI
│   ├── init.py
│   ├── main.py
│   ├── api/
│   │   ├── init.py
│   │   └── endpoints.py
│   └── models/
│       ├── init.py
│       └── inference.py
├── data/bangla_taka.yaml
├── train_yolo.sh
├── requirements.txt
├── Dockerfile
├── .gitignore
└── README.md
```
## Setup
1. Create/activate venv: `python -m venv venv && source venv/bin/activate`
2. Install: `pip install -r requirements.txt`

## Prepare Dataset
`python src/dataset.py`

## Train Model
`./train_yolo.sh`

## Run API
`python app/main.py`

- Docs: http://localhost:8000/docs
- Predict: POST /predict with image file.

## Test
`yolo task=detect mode=predict model=weights/bangla_taka_yolov8.pt source=data/training/takas.png`

## Docker
`docker build -t notem . && docker run -p 8000:8000 notem`

## License
MIT