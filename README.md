# 🚦 Traffic Violation Detection & Localization System

[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.3%2BCUDA-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![YOLO11](https://img.shields.io/badge/YOLO-11-00FFFF?style=for-the-badge&logo=yolo&logoColor=black)](https://ultralytics.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![CUDA](https://img.shields.io/badge/NVIDIA%20RTX%204050-GPU%20Accelerated-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

A deep learning computer vision system designed to detect and localize common traffic violations—specifically **riders without helmets** and **vehicle overloading**—along with safety compliance verification (**helmet wearing**).

This project features a **Dual-Engine Architecture**:
1. **Object Detection Engine (`yolo11n.pt`)**: Detects and plots precise spatial **bounding boxes** around individual riders, helmets, and overloaded vehicles with class labels and confidence percentages.
2. **Classification Engine (`yolo11n-cls.pt`)**: Performs whole-image classification, confidence distribution analysis, and blur/image quality filtering.
3. **Interactive Web Application (`app.py`)**: A modern, responsive FastAPI Single-Page Application (SPA) with real-time drag-and-drop inference, clipboard paste support, dynamic confidence thresholding, one-click sample testing, and REST APIs.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [System Architecture](#-system-architecture)
- [Repository File Structure](#-repository-file-structure)
- [Technologies & Hardware](#-technologies--hardware)
- [Dataset Architecture & Breakdown](#-dataset-architecture--breakdown)
- [Dual-Engine Models](#-dual-engine-models)
  - [1. Object Detection Engine (Bounding Boxes)](#1-object-detection-engine-bounding-boxes)
  - [2. Image Classification Engine](#2-image-classification-engine)
- [Experimental Results & Benchmarks](#-experimental-results--benchmarks)
- [Interactive Web Application (`app.py`)](#-interactive-web-application-apppy)
- [API Endpoints Reference](#-api-endpoints-reference)
- [Installation & Quick Start](#-installation--quick-start)
- [Troubleshooting & Windows Optimization](#-troubleshooting--windows-optimization)
- [Future Roadmap](#-future-roadmap)
- [Author & Acknowledgments](#-author--acknowledgments)

---

## 📌 Overview

Road safety violations such as two-wheeler riders traveling without protective helmets and dangerous vehicle overloading are primary contributors to urban traffic fatalities. Manual enforcement by traffic police at busy intersections is labor-intensive and prone to human error.

This system provides automated visual traffic monitoring:
* **Real-time localization**: Pinpoints multiple riders and helmets in busy urban street scenes.
* **Instant violation alerting**: Color-coded categorization for immediate law enforcement review.
* **Edge deployment ready**: Sub-5ms inference per image using GPU acceleration.

---

## 🧠 System Architecture

```text
                               Traffic Image / Camera Stream
                                             │
                                             ▼
                      ┌──────────────────────────────────────────────┐
                      │            FastAPI Server (app.py)           │
                      └──────────────────────┬───────────────────────┘
                                             │
                    ┌────────────────────────┴────────────────────────┐
                    │                                                 │
                    ▼                                                 ▼
     ┌─────────────────────────────┐                   ┌─────────────────────────────┐
     │   Object Detection Engine   │                   │    Classification Engine    │
     │  (helmet_detection_model)   │                   │   (traffic_violation_model) │
     └──────────────┬──────────────┘                   └──────────────┬──────────────┘
                    │                                                 │
                    ▼                                                 ▼
      Spatial Coordinates [x1,y1,x2,y2]                  Global Class Probability:
      - 🪖 helmet                                        - helmet (99.0%)
      - 🚨 no_helmet                                     - no_helmet
      - 🚛 overloading                                   - overloading
                    │                                    - blur
                    │                                                 │
                    └────────────────────────┬────────────────────────┘
                                             │
                                             ▼
                             Visual Feedback & Web Dashboard:
                             - Bounding Boxes Plotted on Canvas
                             - Status Badges & Localized Box Table
                             - Confidence Distribution Metrics
```

---

## 📂 Repository File Structure

```text
Traffic Violation Detection-Deep Learning/
│
├── saved_models/                                  # Production-ready trained model weights
│   ├── helmet_detection_model.pt                  # YOLO11 Object Detection weights (bounding box head)
│   ├── helmet_model.pt                            # YOLO11 Classification weights (backward compatibility)
│   └── traffic_violation_model.pt                 # YOLO11 Classification weights (aliased)
│
├── Traffic Violations Dataset/                    # Multi-split image dataset
│   ├── train/                                     # Training split (1,799 images)
│   │   ├── helmet/                                # 601 images
│   │   ├── no_helmet/                             # 594 images
│   │   └── overloading/                           # 604 images
│   ├── validation/                                # Validation split (300 images)
│   │   ├── helmet/                                # 100 images
│   │   ├── no_helmet/                             # 100 images
│   │   └── overloading/                           # 100 images
│   ├── test/                                      # Final evaluation split (297 images)
│   │   ├── helmet/                                # 100 images
│   │   ├── no_helmet/                             # 54 images
│   │   └── overloading/                           # 143 images
│   ├── train.cache                                # Ultralytics fast RAM cache
│   ├── validation.cache                           # Validation cache
│   └── test.cache                                 # Test cache
│
├── runs/                                          # Training and evaluation logs & curves
│   ├── detect/                                    # Object detection runs
│   │   └── Traffic_Detection_Runs/
│   │       ├── helmet_detection-5/                # Checkpoints, PR curves, F1 curve, confusion matrix
│   │       │   ├── weights/                       # best.pt and last.pt checkpoints
│   │       │   ├── BoxPR_curve.png                # Precision-Recall curve
│   │       │   ├── BoxF1_curve.png                # F1 confidence curve
│   │       │   ├── results.png                    # Training/validation losses across epochs
│   │       │   └── results.csv                    # Loss and mAP50 telemetry
│   │       └── ...                                # Previous detection iterations (1 to 4)
│   │
│   └── classify/                                  # Classification runs
│       ├── Traffic_Violation_Runs/
│       │   └── helmet_classification/             # 15-epoch classification run artifacts
│       │       ├── weights/                       # best.pt and last.pt
│       │       └── results.csv                    # Accuracy metrics
│       └── val/                                   # Evaluation results on unseen test set
│           ├── confusion_matrix.png               # Test confusion matrix
│           └── confusion_matrix_normalized.png    # Normalized confusion matrix
│
├── app.py                                         # FastAPI web application & interactive SPA interface
├── videodetection.py                              # Real-time video object detection with HUD & export
├── run_app.bat                                    # One-click Windows launcher for web app
├── run_video.bat                                  # One-click Windows launcher for video detection
├── v1.mp4                                         # Sample traffic video for detection testing
├── requirements.txt                               # Project package dependencies
├── training.ipynb                                 # Jupyter Notebook for classification model training & test
├── source.txt                                     # Kaggle dataset provenance & source references
├── .gitignore                                     # Git exclusion rules for large models & caches
├── yolo11n-cls.pt                                 # Pretrained YOLO11 Nano classification backbone
├── yolo26n.pt                                     # Pretrained COCO general detection backbone
└── README.md                                      # Comprehensive project documentation
```

---

## 🛠️ Technologies & Hardware

| Component | Specification / Tool | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.13 | Primary development environment |
| **Framework** | Ultralytics YOLO11 | Deep learning architecture & training |
| **Deep Learning** | PyTorch 2.13.0 + CUDA 13.2 | Neural network tensor computation & autograd |
| **Acceleration** | NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM) | Model training and ultra-low latency inference |
| **Web Server** | FastAPI & Uvicorn | High-performance asynchronous API backend |
| **Frontend UI** | HTML5 / CSS3 / Vanilla JavaScript (SPA) | Drag-and-drop dashboard, canvas rendering |
| **Computer Vision** | OpenCV (`cv2`) & Pillow (`PIL`) | Image decoding, BGR-to-RGB conversion, drawing |
| **Data Science** | NumPy, Pandas, Matplotlib | Telemetry logging, metric curves, confusion matrices |

---

## 📊 Dataset Architecture & Breakdown

The primary dataset is structured into **Train**, **Validation**, and **Test** splits across traffic compliance and violation categories:

```text
Traffic Violations Dataset: 2,396 Total Images
```

| Class | Train Set | Validation Set | Test Set | Total Images | Traffic Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| 🪖 **`helmet`** | 601 | 100 | 100 | **801** | ✅ Compliant / Safe |
| 🚨 **`no_helmet`** | 594 | 100 | 54 | **748** | 🚨 Traffic Violation |
| 🚛 **`overloading`** | 604 | 100 | 143 | **847** | 🚨 Traffic Violation |
| **Total** | **1,799** | **300** | **297** | **2,396** | — |

*Source Reference:* Curated traffic datasets available on Kaggle (see [source.txt](file:///c:/My%20Space/Github_Repo/Traffic%20Violation%20Detection-Deep%20Learning/source.txt)).

---

## 🤖 Dual-Engine Models

### 1. Object Detection Engine (Bounding Boxes)
* **Checkpoint File**: `saved_models/helmet_detection_model.pt`
* **Base Architecture**: `yolo11n.pt` (Object Detection Head)
* **Task Type**: `detect`
* **Classes**: `{0: 'helmet', 1: 'no_helmet', 2: 'overloading'}`
* **Inference Method**: Computes bounding box coordinates `[x1, y1, x2, y2]`, class index, and confidence score. Renders overlays using `result.plot(line_width=3, font_size=1)`.

### 2. Image Classification Engine
* **Checkpoint File**: `saved_models/traffic_violation_model.pt` (aliased to `helmet_model.pt`)
* **Base Architecture**: `yolo11n-cls.pt` (Classification Head)
* **Task Type**: `classify`
* **Classes**: `{0: 'blur', 1: 'helmet', 2: 'no_helmet', 3: 'overloading'}`
* **Inference Method**: Produces global class likelihood vector via Softmax across the entire frame.

---

## 📈 Experimental Results & Benchmarks

### Classification Performance (Trained in `training.ipynb`)
Evaluated across 15 training epochs and tested against the 297 unseen test images:

* **Validation Top-1 Accuracy**: **`99.00%`** (Epoch 15)
* **Final Test Top-1 Accuracy**: **`81.48%`**
* **Final Test Top-5 Accuracy**: **`100.00%`**
* **Inference Speed**: **0.4 ms** preprocess, **4.7 ms** inference per image on NVIDIA RTX 4050 GPU.

### Object Detection Performance (`helmet_detection-5`)
* **mAP@50**: **`30.7%`** (Initial 3-epoch transfer run)
* **Precision / Recall**: 34.1% Precision / 28.0% Recall
* **Average Detection Latency**: **~15 - 25 ms** per 640×640 image frame on GPU.

---

## 🌐 Interactive Web Application (`app.py`)

A full-stack, real-time web application is included to interact with both detection and classification engines without external UI libraries.

### Key Features
1. **Interactive Drag-and-Drop Dropzone**:
   - Drag and drop traffic images directly from file manager.
   - Click to browse local files.
   - **Clipboard Paste (`Ctrl + V`)**: Copy any screenshot or web image and paste directly onto the dashboard.
2. **Real-Time Bounding Box Overlays**:
   - Distinct color-coded bounding boxes drawn directly on canvas:
     - 🪖 **Green**: Helmet (Compliant)
     - 🚨 **Red**: No Helmet (Violation)
     - 🚛 **Dark Red**: Overloading (Violation)
3. **Interactive Sensitivity Controls**:
   - Dynamic **Detection Confidence Slider** (`5%` to `80%`, default `15%`).
   - "Boxes vs. Original" toggle to compare raw vs. annotated images with zero latency.
4. **Detailed Bounding Box Telemetry**:
   - Lists every detected box with exact pixel bounding coordinates `[x1, y1, x2, y2]` and confidence tags.
   - Instant count breakdown: Total Targets, Violations, Compliant Riders.
5. **One-Click Curated Test Samples**:
   - Pre-loaded test samples accessible with a single click:
     - 🪖 **Helmet (Safe)** — *Test helmmet (19).jpg* (**87.8%** conf)
     - 🚨 **No Helmet (Violation)** — *test-nohelmet (3).jpg* (**82.4%** conf)
     - 🚨 **Multi-Riders Without Helmet** — *test-nohelmet (4).jpg* (**3 boxes detected**)
     - 🚛 **Overloading** — *test-nohelmet (28).jpg* (**79.0%** conf)
6. **Dual Mode Switcher**:
   - Seamlessly toggle between **Object Detection** (Bounding Boxes) and **Classification** (Whole-Image Probabilities).

---

## 🔌 API Endpoints Reference

The FastAPI server provides automated Swagger interactive documentation at `http://127.0.0.1:8000/docs`.

| Method | Endpoint | Description | Parameters / Payload |
| :--- | :--- | :--- | :--- |
| `GET` | `/` | Web Application Dashboard | None (returns HTML SPA) |
| `GET` | `/api/info` | Hardware & Model Telemetry | Returns active GPU, PyTorch version, loaded weights |
| `POST` | `/api/predict` | Run Model Inference | `file` (multipart), `mode` (`detect` or `classify`), `conf` (float) |
| `POST` | `/api/predict-url` | Predict from Remote Link | `url` (str), `mode` (`detect` or `classify`), `conf` (float) |
| `GET` | `/api/samples` | List Pre-loaded Samples | Returns sample metadata and preview endpoints |
| `GET` | `/api/sample-file` | Stream Sample Image | `id` (sample query string) |

### Sample `POST /api/predict` Response (Detection Mode):
```json
{
  "mode": "detect",
  "status": "TRAFFIC VIOLATION DETECTED",
  "status_sub": "2 violation box(es) detected",
  "status_color": "#ef4444",
  "status_icon": "🚨",
  "total_detections": 2,
  "violation_count": 2,
  "compliant_count": 0,
  "boxes": [
    {
      "class": "no_helmet",
      "label": "No Helmet",
      "confidence_pct": 82.4,
      "bbox": [779.1, 292.2, 1050.6, 826.5],
      "color": "#ef4444",
      "type": "violation"
    }
  ],
  "annotated_image": "data:image/jpeg;base64,...",
  "inference_ms": 18.4,
  "device": "NVIDIA GeForce RTX 4050 Laptop GPU",
  "image_size": "1080 × 1920"
}
```

---

## 🚀 Installation & Quick Start

### 1. Prerequisites
- Python 3.10+ (Python 3.13 recommended)
- NVIDIA GPU with CUDA 12.x or 13.x (Optional, falls back to CPU automatically)

### 2. Clone Repository
```bash
git clone https://github.com/NishantRajora/Traffic-Violation-Detection-Deep-Learning.git
cd "Traffic Violation Detection-Deep Learning"
```

### 3. Install Dependencies
```bash
# Optional: For NVIDIA GPU acceleration (CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install all project dependencies
pip install -r requirements.txt
```

### 4. Run the Web Application
* **Windows (One-Click)**:
  Double-click [run_app.bat](file:///c:/My%20Space/Github_Repo/Traffic%20Violation%20Detection-Deep%20Learning/run_app.bat)

* **Terminal**:
  ```bash
  python app.py
  ```

* **Open in Browser**:
  Navigate to **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

### 5. Run Video Object Detection (`videodetection.py`)
Run real-time bounding box detection on video files (like `v1.mp4`) or live webcams:

* **Windows (One-Click)**:
  Double-click [run_video.bat](file:///c:/My%20Space/Github_Repo/Traffic%20Violation%20Detection-Deep%20Learning/run_video.bat)

* **Interactive Mode**:
  ```bash
  python videodetection.py
  ```
  *(Prompts you to select `v1.mp4`, a custom file, or webcam, plus set the confidence threshold)*

* **CLI Direct Execution (Sample `v1.mp4`)**:
  ```bash
  python videodetection.py --source v1.mp4 --conf 0.15
  ```

* **CLI Live Playback (Sample `v1.mp4`)**:
  ```bash
  python videodetection.py --source v1.mp4 --conf 0.15
  ```
  *(Displays real-time playback window with bounding boxes, violation HUD, and telemetry. Press `q` to quit, `space` to pause/resume. No video files are saved to disk)*

* **Optional: Save to Disk**:
  If you ever want to save an annotated video file, add the `--save` flag:
  ```bash
  python videodetection.py --source v1.mp4 --conf 0.15 --save
  ```

---

## 🔧 Troubleshooting & Windows Optimization

### OpenMP Conflict (`libiomp5md.dll`)
On Windows environments with multiple OpenMP runtimes (e.g., Anaconda + PyTorch), the application automatically injects:
```python
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
```
If executing scripts from PowerShell manually, set:
```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
```

### Multi-Processing Workers in DataLoader
On Windows Jupyter Notebooks, set `workers=0` during training in `model.train(..., workers=0)` to avoid deadlocks.

---

## 🔮 Future Roadmap

- [ ] **Extended Bounding-Box Detection**: Train `yolo11m.pt` for 50+ epochs on high-resolution annotated traffic video datasets.
- [ ] **Live RTSP Stream Processing**: Direct ingestion from intersection CCTV cameras.
- [ ] **Automated Number Plate Recognition (ANPR / ALPR)**: Extract vehicle registration numbers of violating vehicles automatically.
- [ ] **Automated E-Challan Issuance**: Database integration for automated traffic fine generation.

---

## 👨‍💻 Author & Acknowledgments

**Nishant Rajora**  
*B.Tech in Computer Science & Engineering (Specialization: Data Science)*  
**The NorthCap University, Gurugram**

*Special thanks to the Ultralytics team for YOLO11, and the open-source computer vision community.*
