# 🪖 Real-Time Helmet Detection System (YOLOv8 & Supervision)

[![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.13.0%2BCUDA-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?style=for-the-badge&logo=yolo&logoColor=black)](https://ultralytics.com)
[![Supervision](https://img.shields.io/badge/Roboflow-Supervision-6706CE?style=for-the-badge)](https://supervision.roboflow.com)
[![CUDA](https://img.shields.io/badge/NVIDIA%20RTX%204050-GPU%20Accelerated-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)

An end-to-end deep learning computer vision system designed to detect whether two-wheeler riders are wearing helmets (**`With Helmet`**) or committing a safety violation (**`Without Helmet`**) with precise spatial bounding boxes and confidence scores.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [System Architecture](#-system-architecture)
- [Dataset Architecture & Breakdown](#-dataset-architecture--breakdown)
- [Experimental Results & Benchmarks](#-experimental-results--benchmarks)
- [Repository File Structure](#-repository-file-structure)
- [Installation & Setup](#-installation--setup)
- [Model Training (`train_detection.py`)](#-model-training-train_detectionpy)
- [Inference & Detection (`detect_image.py`)](#-inference--detection-detect_imagepy)
- [Notebook Pipeline (`helmet_detection.ipynb`)](#-notebook-pipeline-helmet_detectionipynb)
- [Troubleshooting & Windows Optimization](#-troubleshooting--windows-optimization)

---

## 📌 Overview

Riding two-wheelers without protective helmets is a major cause of urban road fatalities. This project automates traffic safety monitoring:
* **High-Accuracy Detection**: Detects multiple riders in complex traffic scenes.
* **Instant Safety Classification**: Distinguishes riders with helmets (Green) and violators without helmets (Red).
* **Real-Time Edge Inference**: Operates at sub-10ms per frame on modern NVIDIA GPUs.

---

## 🧠 System Architecture

```text
                               Traffic Image / Video Feed
                                           │
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │      YOLOv8 Object Detector (best.pt)        │
                    └──────────────────────┬───────────────────────┘
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    │                                             │
                    ▼                                             ▼
          🟢 With Helmet (Safe)                    🚨 Without Helmet (Violation)
          - Class 0                                - Class 1
          - Confidence Percentage                  - Confidence Percentage
          - Spatial Bounding Box [x1,y1,x2,y2]     - Spatial Bounding Box [x1,y1,x2,y2]
                    │                                             │
                    └──────────────────────┬──────────────────────┘
                                           │
                                           ▼
                             Visual HUD & Alert Dashboard:
                             - Bounding Boxes Plotted on Canvas
                             - Color-Coded Overlay & Live Count
```

---

## 📂 Dataset Architecture & Breakdown

The dataset originates from Pascal VOC XML annotations and is preprocessed into standard YOLO format:

1. **Raw Source (`data/`)**:
   * Total Images: **764** images (`data/images/`)
   * Annotations: **764** XML files (`data/annotations/`)

2. **YOLO Splits (`HelmetDataset/`)**:
   * **Train Split (80%)**: 611 images (`HelmetDataset/train/`)
   * **Validation Split (10%)**: 76 images (`HelmetDataset/valid/`)
   * **Test Split (10%)**: 77 images (`HelmetDataset/test/`)

3. **Classes (`nc: 2`)**:
   * `0`: **With Helmet**
   * `1`: **Without Helmet**

Configuration (`HelmetDataset/data.yaml`):
```yaml
path: HelmetDataset
train: train/images
val: valid/images
test: test/images
nc: 2
names:
  - With Helmet
  - Without Helmet
```

---

## 📊 Experimental Results & Benchmarks

Trained on **NVIDIA GeForce RTX 4050 Laptop GPU** across 20 epochs using `yolov8n.pt`:

| Evaluation Split | Class | Images | Instances | Precision (P) | Recall (R) | mAP@50 | mAP@50-95 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Validation** | **All Classes** | **74** | **141** | **78.0%** | **86.5%** | **88.0%** | **54.4%** |
| Validation | With Helmet | 48 | 86 | 76.4% | 88.4% | **90.5%** | 61.5% |
| Validation | Without Helmet | 33 | 55 | 79.5% | 84.5% | **85.5%** | 47.3% |
| **Test (Unseen)** | **All Classes** | **76** | **159** | **76.2%** | **80.6%** | **81.5%** | **48.6%** |
| Test (Unseen) | With Helmet | 53 | 97 | 81.1% | 93.0% | **87.2%** | 54.0% |
| Test (Unseen) | Without Helmet | 27 | 62 | 71.3% | 68.1% | **75.7%** | 43.1% |

* **Inference Speed**: ~8.3 ms per image (~120 FPS on RTX 4050 GPU).

---

## 📂 Repository File Structure

```text
Traffic Violation Detection-Deep Learning/
│
├── saved_models/                                  # Exported production model weights
│   └── best.pt                                    # Fine-tuned YOLOv8 helmet detection model
│
├── HelmetDataset/                                 # YOLO-formatted dataset splits
│   ├── train/                                     # 611 training images & txt labels
│   ├── valid/                                     # 76 validation images & txt labels
│   ├── test/                                      # 77 test images & txt labels
│   └── data.yaml                                  # Ultralytics dataset configuration
│
├── data/                                          # Raw dataset
│   ├── images/                                    # 764 original road images
│   └── annotations/                               # 764 Pascal VOC XML files
│
├── runs/detect/                                   # Training and validation artifacts
│   ├── train/                                     # 20-epoch training curves, confusion matrix
│   │   ├── weights/best.pt                        # Best checkpoint
│   │   ├── results.png                            # Loss and mAP progression plots
│   │   └── results.csv                            # Epoch-by-epoch telemetry
│   └── val/                                       # Test evaluation curves and visual batch predictions
│
├── train_detection.py                             # Modular training script for YOLOv8
├── detect_image.py                                # Inference script for images, folders & video
├── split_data.ipynb                               # Step 1: Dataset preparation & splitting notebook
├── train_model.ipynb                              # Step 2: YOLOv8 model training & saving notebook
├── test_model.ipynb                               # Step 3: Model evaluation, testing & visual inference
├── helmet_detection.ipynb                         # Complete all-in-one pipeline notebook
├── requirements.txt                               # Project package dependencies
├── .gitignore                                     # Exclusion rules for datasets, weights & caches
└── README.md                                      # Project documentation
```

---

## 🚀 Installation & Setup

### 1. Clone & Navigate to Repository
```bash
git clone https://github.com/NishantRajora/Traffic-Violation-Detection-Deep-Learning.git
cd "Traffic Violation Detection-Deep Learning"
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🏋️ Model Training (`train_detection.py`)

To train or fine-tune the model:

* **Default Training (20 epochs, batch 16, GPU acceleration)**:
  ```bash
  python train_detection.py
  ```

* **Custom Hyperparameters**:
  ```bash
  python train_detection.py --epochs 30 --batch 16 --imgsz 640
  ```

* **Convert Raw Pascal VOC XML Dataset**:
  If you have raw images and XML annotations in `data/`, convert them to YOLO format using:
  ```bash
  python train_detection.py --convert-voc
  ```

---

## 🔍 Inference & Detection (`detect_image.py`)

Run inference on test images, image folders, or live video feeds:

* **Detect Single Image**:
  ```bash
  python detect_image.py --source HelmetDataset/test/images/BikesHelmets10.png
  ```

* **Detect and Save Annotated Output**:
  ```bash
  python detect_image.py --source HelmetDataset/test/images/BikesHelmets10.png --save
  ```
  *(Output is saved to `output/detected_BikesHelmets10.png`)*

* **Detect on Entire Directory of Images**:
  ```bash
  python detect_image.py --source HelmetDataset/test/images --save
  ```

* **Detect on Webcam (Live)**:
  ```bash
  python detect_image.py --source 0
  ```

* **Detect on Video File**:
  ```bash
  python detect_image.py --source video.mp4 --show
  ```

---

## 📓 Modular Notebook Pipeline

The workflow has been split into 3 clear, dedicated Jupyter Notebooks:

1. **`split_data.ipynb`**:
   * Parses Pascal VOC XML annotations from `data/` using `supervision`.
   * Splits dataset into **Train (80%)**, **Validation (10%)**, and **Test (10%)**.
   * Exports to normalized YOLO format under `HelmetDataset/`.
   * Configures and validates `HelmetDataset/data.yaml`.
   * Verifies file counts and visualizes random training images.

2. **`train_model.ipynb`**:
   * Configures GPU hardware and Windows OpenMP settings.
   * Loads base YOLOv8 model (`yolov8n.pt`).
   * Trains model for 20 epochs on `HelmetDataset/data.yaml`.
   * **Saves the best model** to `saved_models/best.pt`.
   * Inspects `results.csv` and plots loss curves and mAP progression over epochs.

3. **`test_model.ipynb`**:
   * Loads the trained model from `saved_models/best.pt`.
   * Evaluates on unseen **Test Dataset (77 images)** using `model.val(split='test')`.
   * Displays Confusion Matrix, PR curve, and F1 curve.
   * Visualizes predictions on random test images with bounding boxes and confidence.
   * Provides single-image violation detection and video inference demo.

---

## 🔧 Troubleshooting & Windows Optimization

### OpenMP Duplicate Runtime Conflict (`libiomp5md.dll`)
On Windows environments with multiple OpenMP runtimes, the scripts include:
```python
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
```

### Multi-Processing Workers in DataLoader
On Windows systems, set `workers=0` in training scripts to prevent multiprocessing deadlocks.
