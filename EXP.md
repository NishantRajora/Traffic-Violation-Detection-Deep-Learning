# 📘 Complete Technical Reference & Deep-Dive Explanation (`expalin.md`)
## AI-Powered Traffic Violation Detection System (YOLOv8, Supervision & FastAPI)

This document provides an exhaustive, end-to-end technical reference for the **Traffic Violation Detection Deep Learning** project. It covers the complete data flow, every Jupyter Notebook (`.ipynb`) and Python script (`.py`), exact experimental results, head-to-head model comparisons, and rigorous mathematical definitions of every metric used.

---

## 📑 Table of Contents
1. [End-to-End Project Flow & Architecture](#1-end-to-end-project-flow--architecture)
2. [Core Definitions & Mathematical Formulas (All Metrics Explained)](#2-core-definitions--mathematical-formulas-all-metrics-explained)
3. [Stage 1: Data Engineering & Splitting (`split_data.ipynb`)](#3-stage-1-data-engineering--splitting-split_dataipynb)
4. [Stage 2: Model Training & Telemetry (`train_model.ipynb` & `runs/detect/train/`)](#4-stage-2-model-training--telemetry-train_modelipynb--runsdetecttrain)
5. [Stage 3: Model Testing & Evaluation (`test_model.ipynb` & `evaluate_all_metrics.py`)](#5-stage-3-model-testing--evaluation-test_modelipynb--evaluate_all_metricspy)
6. [Stage 4: Head-to-Head Model Comparison (`compare.ipynb` & `compre.py`)](#6-stage-4-head-to-head-model-comparison-compareipynb--comprepy)
7. [Stage 5: Production Web Dashboard (`app.py`)](#7-stage-5-production-web-dashboard-apppy)
8. [Complete Repository File & Artifact Map](#8-complete-repository-file--artifact-map)

---

## 1. End-to-End Project Flow & Architecture

The project transforms raw, unsplit road surveillance images into a real-time, production-ready FastAPI web application through **5 sequential engineering stages**:

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: DATA ENGINEERING (split_data.ipynb)                                │
│   Raw Data: 764 Images (.png) + 764 Pascal VOC Annotations (.xml)           │
│   Action:   Convert VOC [xmin, ymin, xmax, ymax] -> Normalized YOLO .txt    │
│   Output:   HelmetDataset/ (80% Train: 611 | 10% Val: 76 | 10% Test: 77)    │
│             + HelmetDataset/data.yaml                                       │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: DEEP LEARNING TRAINING (train_model.ipynb)                         │
│   Input:    yolov8n.pt (Pretrained Backbone) + HelmetDataset/data.yaml      │
│   Action:   Fine-tune YOLOv8 Nano (imgsz=640, batch=16) on GPU/CUDA         │
│   Output:   saved_models/best.pt (5.95 MB, 3,011,238 parameters)            │
│             + runs/detect/train/ (results.csv, results.png, labels.jpg)     │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: UNSEEN TEST EVALUATION (test_model.ipynb & evaluate_all_metrics.py)│
│   Input:    saved_models/best.pt + HelmetDataset/test/ (Unseen Images)      │
│   Action:   Run Ultralytics model.val(split='test') + Visual Inference      │
│   Output:   mAP@50: 94.90% | Precision: 91.28% | Recall: 90.14% | F1: 90.71%│
│             + Confusion Matrix, PR Curve, F1 Curve                          │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 4: HEAD-TO-HEAD BENCHMARKING (compare.ipynb & compre.py)              │
│   Models:   Model A (Custom best.pt) vs. Model B (helmet_detection_model.pt)│
│   Action:   Direct Pixel-Level IoU Matching across 77 Test Images (139 GT)  │
│   Output:   100-Point Decision Matrix -> Model A wins 65.0 to 35.0          │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 5: PRODUCTION DEPLOYMENT (app.py)                                     │
│   Backend:  FastAPI + Uvicorn + OpenCV + yt-dlp                             │
│   Inputs:   Image Upload, Sample Gallery, Image URL, Video Upload,          │
│             YouTube Stream, Live Webcam Frames                              │
│   Output:   Real-time Bounding Box HUD, Model Switcher, CSV History Export  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Core Definitions & Mathematical Formulas (All Metrics Explained)

### 2.1 Bounding Box Coordinate Formats
1. **Pascal VOC Format (Raw `.xml`):**
   Stores absolute pixel coordinates of the top-left and bottom-right corners:
   $$\text{Box}_{\text{VOC}} = [x_{\min}, y_{\min}, x_{\max}, y_{\max}]$$
2. **YOLO Format (Normalized `.txt`):**
   Stores values normalized between $0.0$ and $1.0$ relative to image width $W$ and height $H$:
   $$x_c = \frac{x_{\min} + x_{\max}}{2 \cdot W}, \quad y_c = \frac{y_{\min} + y_{\max}}{2 \cdot H}, \quad w = \frac{x_{\max} - x_{\min}}{W}, \quad h = \frac{y_{\max} - y_{\min}}{H}$$
   - **Why normalize?** So the neural network can process images of any resolution consistently without depending on absolute pixel sizes.

### 2.2 Intersection over Union (IoU)
**Definition:** Measures how tightly a predicted bounding box ($B_p$) overlaps with the human-labeled ground-truth box ($B_{gt}$).
$$\text{IoU} = \frac{\text{Area}(B_p \cap B_{gt})}{\text{Area}(B_p \cup B_{gt})}$$
- $\text{IoU} = 1.0$: Perfect 100% overlap.
- $\text{IoU} = 0.0$: No overlap at all.
- **Standard Threshold ($\text{IoU} \ge 0.50$):** If a predicted box overlaps at least 50% with the ground truth and has the matching class label, it is counted as a **True Positive (TP)**.

### 2.3 Confusion Matrix Outcomes in Traffic Enforcement
| Term | Technical Definition | Real-World Traffic Enforcement Meaning |
| :--- | :--- | :--- |
| **True Positive (TP)** | Model predicts a box that matches a ground-truth box ($\text{IoU} \ge 0.50$) and class. | Correctly identifies a rider with a helmet, or correctly catches a violator without a helmet. |
| **False Positive (FP)** | Model predicts a box where no matching ground truth exists (or wrong class / $\text{IoU} < 0.50$). | **False Alarm / False Citation:** E.g., ticketing an innocent rider who was actually wearing a helmet, or mistaking a headlight/bag for a head. |
| **False Negative (FN)** | A ground-truth object exists in the image, but the model fails to detect it. | **Missed Violator:** E.g., an unhelmeted rider drives past the camera completely undetected. |

### 2.4 Precision, Recall, and F1-Score
* **Precision ($P$):** *"Out of everything the model flagged, how much was actually right?"*
  $$P = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
  - High Precision minimizes **False Citations** (unfair fines).
* **Recall ($R$):** *"Out of all real riders/violators in the scene, how many did the model catch?"*
  $$R = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
  - High Recall minimizes **Missed Violators** (escaped offenders).
* **F1-Score:** The harmonic mean of Precision and Recall. It penalizes extreme imbalances (e.g., high recall with terrible precision).
  $$F_1 = 2 \cdot \frac{P \cdot R}{P + R}$$

### 2.5 Average Precision (AP) & Mean Average Precision (mAP)
* **AP (Average Precision):** The area under the Precision-Recall curve for a single class across all confidence thresholds ($0.0$ to $1.0$).
* **$\text{mAP}@50$ (mAP at $\text{IoU} = 0.50$):** The mean of AP across all classes (`With Helmet` and `Without Helmet`) evaluated at $\text{IoU} \ge 0.50$. **This is the primary industry-standard "Accuracy" metric for object detection.**
* **$\text{mAP}@50\text{-}95$:** The average mAP computed across 10 strict IoU thresholds: $0.50, 0.55, 0.60, \dots, 0.95$. This measures how razor-sharp and tight the bounding boxes are.

### 2.6 YOLOv8 Training Loss Functions
During training (`results.csv`), YOLOv8 minimizes three loss components simultaneously:
1. **`box_loss` (CIoU Loss):** Complete IoU loss; penalizes center-point distance, overlap area, and aspect ratio mismatch between predicted and target boxes.
2. **`cls_loss` (Binary Cross-Entropy Loss):** Penalizes wrong class probability predictions (`With Helmet` vs. `Without Helmet`).
3. **`dfl_loss` (Distribution Focal Loss):** Refines bounding box edge localization when boundaries are blurry or partially occluded.

### 2.7 Speed & Real-Time Performance Metrics
* **Latency ($\text{ms}$):** Average time taken to process a single image/frame (Preprocess + Model Inference + Postprocess/NMS).
* **Jitter / Std Dev ($\text{ms}$):** Standard deviation of latency across frames. Low jitter means smooth, predictable video streaming.
* **Throughput ($\text{FPS}$):** Frames Per Second:
  $$\text{FPS} = \frac{1000}{\text{Latency (ms)}}$$
  Standard CCTV feeds run at **30 FPS** ($\le 33.3\text{ ms}$ per frame).

### 2.8 Confidence Score
A probability value in $[0.0, 1.0]$ output by the model for each bounding box indicating how certain it is that the box contains that class. Predictions below the `conf` threshold (typically `0.25` or `0.30`) are discarded.

---

## 3. Stage 1: Data Engineering & Splitting (`split_data.ipynb`)

### Implementation Details
- **Source Directory:** `data/images/` (764 `.png` files) and `data/annotations/` (764 Pascal VOC `.xml` files).
- **Library Used:** `supervision` (`sv.DetectionDataset.from_pascal_voc`).
- **Split Logic:**
  1. `train_ds, rem_ds = dataset.split(split_ratio=0.8)` $\rightarrow$ **611 Training Images (80%)**
  2. `val_ds, test_ds = rem_ds.split(split_ratio=0.5)` $\rightarrow$ **76 Validation Images (10%)** and **77 Test Images (10%)**
- **Export:** `ds.as_yolo(...)` writes normalized `.txt` label files into `HelmetDataset/{train,val,test}/labels/`.
- **Manifest Generation (`HelmetDataset/data.yaml`):**
  Uses `FINAL_DIR.as_posix()` to ensure forward slashes (`/`) on Windows so Ultralytics PyYAML parses paths cleanly without escape-character syntax errors.
  ```yaml
  path: C:/My Space/GitHub Repo/Traffic-Violation-Detection-Deep-Learning/HelmetDataset
  train: train/images
  val: val/images
  test: test/images
  nc: 2
  names:
  - With Helmet
  - Without Helmet
  ```

---

## 4. Stage 2: Model Training & Telemetry (`train_model.ipynb` & `runs/detect/train/`)

### Hyperparameters & Configuration
In `train_model.ipynb`, the model is initialized from `yolov8n.pt` (YOLOv8 Nano) and trained with `imgsz=640`, `batch=16`, `device=cuda/cpu`, and `KMP_DUPLICATE_LIB_OK=TRUE` (to prevent OpenMP runtime conflicts on Windows).

From `runs/detect/train/args.yaml`, the actual completed training run executed for **30 epochs** on GPU (`device: '0'`, `batch: 16`, `imgsz: 640`, `amp: true`, `close_mosaic: 10`). The resulting weights were exported to `saved_models/best.pt` (**5.95 MB**, **3,011,238 parameters**).

### Training Telemetry Progression (`runs/detect/train/results.csv`)

| Epoch | Time (s) | Train Box Loss | Train Cls Loss | Train DFL Loss | Val Precision | Val Recall | Val mAP@50 | Val mAP@50-95 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 10.12 | 1.5564 | 2.8495 | 1.2987 | 0.42% | 60.44% | 1.09% | 0.61% |
| **16** | 102.56 | 1.2066 | 0.8758 | 1.0913 | 70.21% | 84.66% | 85.16% | 52.97% |
| **20** | 126.91 | 1.1769 | 0.7717 | 1.0725 | 80.89% | 81.02% | 84.51% | 52.49% |
| **25** | 184.83 | 1.0748 | 0.6642 | 1.0502 | 82.90% | 80.73% | 87.43% | 54.30% |
| **26** *(Peak mAP)* | 190.77 | 1.0565 | 0.6405 | 1.0431 | 81.20% | 80.28% | **87.54%** | **54.70%** |
| **28** *(Peak Prec)* | 202.61 | 1.0423 | 0.5997 | 1.0333 | **84.73%** | 78.28% | 86.78% | 52.73% |
| **30** *(Final)* | 214.62 | 1.0110 | 0.5834 | 1.0181 | 82.47% | 79.25% | 86.68% | 52.51% |

- **Key Observation:** Training completed in **214.62 seconds (~3.6 minutes)** on GPU. `train/cls_loss` dropped smoothly from `2.85` to `0.58`, and `val/mAP@50` climbed from `1.09%` to a peak of **`87.54%`** at Epoch 26 without divergence or overfitting.

---

## 5. Stage 3: Model Testing & Evaluation (`test_model.ipynb` & `evaluate_all_metrics.py`)

Both `test_model.ipynb` and `evaluate_all_metrics.py` run Ultralytics' official validation engine on the unseen test split:
```python
metrics = model.val(data=str(DATA_YAML), split="test")
```

### Exact Test Results (`saved_models/best.pt` on `HelmetDataset/test/`)
*Note on Dataset Integrity:* Out of 77 test images, Ultralytics automatically skips 1 corrupt label (`BikesHelmets444.png`, which contains unnormalized coordinates `[298.9975, 47.99686, 78.005, 80.00629]`), evaluating **76 valid images** containing **136 ground-truth instances** (102 `With Helmet`, 34 `Without Helmet`).

#### Overall Detection Accuracy & Speed (CPU Benchmark: 13th Gen Intel Core i5-13420H)
| Metric | Value | Meaning |
| :--- | :---: | :--- |
| **Detection Accuracy (`mAP@50`)** | **94.90%** | Overall accuracy across confidence thresholds at $\text{IoU} \ge 0.50$ |
| **Strict Accuracy (`mAP@50-95`)** | **64.98%** | Average mAP across IoU thresholds from $0.50$ to $0.95$ |
| **Overall Precision ($P$)** | **91.28%** | 91.28% of all predicted bounding boxes are accurate |
| **Overall Recall ($R$)** | **90.14%** | 90.14% of all real ground-truth targets are captured |
| **Balanced F1-Score** | **90.71%** | Harmonic mean of Precision and Recall |
| **CPU Inference Latency** | **45.10 ms** | Total frame latency: `47.20 ms` (`1.67 ms` pre + `45.10 ms` inf + `0.42 ms` post) $\approx$ **21.2 FPS on CPU** |

#### Per-Class Breakdown
| Class ID | Class Name | Images | Instances | Precision | Recall | F1-Score | mAP@50 | mAP@50-95 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | `With Helmet` | 61 | 102 | **89.17%** | **97.06%** | **92.94%** | **97.48%** | 70.9% |
| **1** | `Without Helmet` | 19 | 34 | **93.40%** | **83.23%** | **88.02%** | **92.32%** | 59.0% |

#### Visual Inference (`test_model.ipynb` Cell 3)
Randomly samples 9 test images (`conf=0.3`, `random.seed(42)`) and displays a $3 \times 3$ annotated grid alongside evaluation curves (`confusion_matrix.png`, `BoxPR_curve.png`, `BoxF1_curve.png`).

---

## 6. Stage 4: Head-to-Head Model Comparison (`compare.ipynb` & `compre.py`)

This stage benchmarks **Model A (Custom Trained: `saved_models/best.pt`)** against **Model B (Off-the-Shelf Pretrained: `saved_models/helmet_detection_model.pt`)**.

### Why `compare.ipynb` Numbers Differ from `test_model.ipynb`
- **`test_model.ipynb` (`model.val()`):** Sweeps across all confidence thresholds ($0.0 \rightarrow 1.0$) to compute the PR-curve area (`mAP`) and skips `BikesHelmets444.png` (evaluating 76 images / 136 instances).
- **`compare.ipynb` (`evaluate_model_direct()`):** Evaluates both models under **identical real-world deployment settings** at a fixed confidence threshold **`conf = 0.25`** and **`iou_thresh = 0.50`** across all **77 test images (139 ground-truth instances: 102 `With Helmet`, 37 `Without Helmet`)** using greedy bipartite IoU matching.

### Complete 8-Dimension Benchmark Results (`compare.ipynb`)

#### 1. Model Architecture & Physical Footprint
| Property | Model A: Custom Trained (`best.pt`) | Model B: Pretrained (`helmet_detection_model.pt`) |
| :--- | :---: | :---: |
| **File Size** | **5.95 MB** | 5.96 MB |
| **Total Parameters** | 3,011,238 | 3,011,238 |
| **Classes (2)** | `With Helmet`, `Without Helmet` | `With helmet`, `Without helmet` |

#### 2. Real-Time Inference Latency & Speed (CPU, 10 warmup + 50 timed runs)
| Metric | Model A: Custom Trained | Model B: Pretrained | Winner |
| :--- | :---: | :---: | :---: |
| **Average Latency** | **52.62 ms** | 60.24 ms | **Model A** (12.6% faster) |
| **Jitter (Std Dev)** | **8.30 ms** | 14.43 ms | **Model A** (more stable) |
| **Throughput (FPS)** | **19.0 FPS** (CPU) | 16.6 FPS (CPU) | **Model A** |

#### 3. Direct Ground-Truth Quantitative Accuracy (77 Test Images, 139 Instances)
| Metric | Model A: Custom Trained | Model B: Pretrained | Winner |
| :--- | :---: | :---: | :---: |
| **Precision** | 82.32% | **82.50%** | Model B (+0.18%) |
| **Recall** | **97.12%** | 94.96% | **Model A (+2.16%)** |
| **F1-Score** | **89.11%** | 88.29% | **Model A (+0.82%)** |
| **Mean IoU** | 83.81% | **85.48%** | Model B (+1.67%) |
| **True Positives (TP)** | **135** | 132 | **Model A (+3 detections)** |
| **False Positives (FP - False Alarms)** | 29 | **28** | Model B (1 fewer FP) |
| **False Negatives (FN - Missed)** | **4** | 7 | **Model A (43% fewer misses)** |

#### 4. Granular Per-Class Breakdown
| Class | Ground Truth | Model A (P / R / F1) | Model B (P / R / F1) | Class Winner |
| :--- | :---: | :---: | :---: | :---: |
| **`With Helmet` (Compliant)** | 102 | 81.5% / **99.0%** / 89.4% | **84.6%** / 97.1% / **90.4%** | Model B |
| **`Without Helmet` (Violator)** | 37 | **85.0%** / **91.9%** / **88.3%** | 76.7% / 89.2% / 82.5% | **Model A (+5.8% F1)** |

* **Why Model A matters for enforcement:** On the critical **`Without Helmet`** violator class, Model A achieves **88.3% F1** (34 TP, 6 FP, 3 FN) vs. Model B's **82.5% F1** (33 TP, 10 FP, 4 FN)—cutting false violation alerts by 40% (6 vs 10 FP) while catching more violators.

#### 5. Average Prediction Confidence (`compre.py`)
- **Model A (Custom):** `0.7117`
- **Model B (Pretrained):** `0.7439`
- *Note:* `compre.py` is a lightweight CLI script that computes the same `evaluate_model_direct` metrics but prints only a simplified verdict based solely on average confidence (`PRETRAINED IS BETTER: 0.7439 vs 0.7117`), whereas `compare.ipynb` uses the full 100-point engineering scorecard below.

#### 6. Automated 100-Point Decision Scorecard (`compare.ipynb` Step 8)
| Evaluation Criterion | Max Points | Model A (Custom) | Model B (Pretrained) | Points Awarded |
| :--- | :---: | :---: | :---: | :---: |
| **Violator Recall (`Without Helmet`)** | 25 pts | **91.9%** | 89.2% | **Model A (+25)** |
| **Overall F1-Score** | 25 pts | **89.11%** | 88.29% | **Model A (+25)** |
| **Precision (False Alarm Avoidance)** | 20 pts | 82.32% | **82.50%** | **Model B (+20)** |
| **Bounding Box Tightness (Mean IoU)** | 15 pts | 83.81% | **85.48%** | **Model B (+15)** |
| **Inference Speed & Latency** | 15 pts | **52.62 ms (19.0 FPS)** | 60.24 ms (16.6 FPS) | **Model A (+15)** |
| **🏆 FINAL COMPOSITE SCORE** | **100 pts** | **65.0 / 100** | **35.0 / 100** | **MODEL A (CUSTOM)** |

---

## 7. Stage 5: Production Web Dashboard (`app.py`)

`app.py` is a full-stack **FastAPI** web server (`uvicorn`) providing an interactive surveillance UI with live model switching and multi-source inference:

### Key Functions & API Endpoints
* **`get_available_models()` & `/get_models` / `/switch_model`:** Dynamically discovers `.pt` models in `saved_models/` (`best.pt`, `helmet_detection_model.pt`) and allows hot-swapping between Model A and Model B at runtime.
* **`classify_detection()` & `annotate_frame()`:** Normalizes class labels across both models (`With Helmet` vs `Without Helmet`) and draws color-coded bounding boxes (Green for Safe/Compliant, Red for Violation) with confidence scores and live HUD counters.
* **`run_pipeline()`:** Executes YOLO inference with configurable `conf` (default `0.25`) and `iou` (default `0.45`), records latency, and logs detection history.
* **Supported Input Endpoints:**
  - `POST /predict`: Static image file upload.
  - `POST /predict_sample`: Runs inference on built-in test samples (`/sample_images`).
  - `POST /predict_url`: Fetches and analyzes remote image URLs.
  - `POST /predict_video_upload`: Processes uploaded video files frame-by-frame.
  - `POST /predict_youtube`: Extracts frames from YouTube videos/streams using `yt-dlp` (`extract_youtube_frame`) and runs detection.
  - `POST /detect_webcam_frame`: Real-time webcam frame inference endpoint.
* **Audit & Reporting Endpoints:**
  - `GET /system_info`: Hardware (GPU/CPU) and model status telemetry.
  - `GET /detection_history`, `POST /clear_history`, `GET /export_history_csv`: View, manage, and download violation audit logs as CSV.

---

## 8. Complete Repository File & Artifact Map

| File / Folder | Role in Project |
| :--- | :--- |
| `data/images/` & `data/annotations/` | Raw dataset: 764 images and 764 Pascal VOC XML files. |
| `split_data.ipynb` | **Step 1:** Converts VOC XMLs to YOLO `.txt` and creates 80/10/10 splits + `data.yaml`. |
| `HelmetDataset/` | Processed YOLO dataset (`train/`: 611, `val/`: 76, `test/`: 77) + `data.yaml`. |
| `train_model.ipynb` | **Step 2:** Fine-tunes `yolov8n.pt` and exports `saved_models/best.pt`. |
| `runs/detect/train/` | Training logs (`results.csv`), hyperparameters (`args.yaml`), loss/mAP plots (`results.png`), and dataset stats (`labels.jpg`). |
| `test_model.ipynb` | **Step 3:** Evaluates `best.pt` on `test` split, prints accuracy/per-class tables, displays curves and 9 random sample predictions. |
| `evaluate_all_metrics.py` | Standalone CLI script printing overall mAP, Precision, Recall, F1, Latency/FPS, per-class breakdown, and confusion matrix. |
| `runs/detect/val*` | Validation artifacts (`confusion_matrix.png`, `BoxPR_curve.png`, `BoxF1_curve.png`, `val_batch*_pred.jpg`). |
| `compare.ipynb` | **Step 4:** Full 8-dimension benchmark & 100-point scorecard comparing `best.pt` vs. `helmet_detection_model.pt`. |
| `compre.py` | Lightweight CLI comparison script evaluating IoU metrics and average confidence. |
| `saved_models/` | Production weights: `best.pt` (Custom, 5.95 MB) and `helmet_detection_model.pt` (Pretrained, 5.96 MB). |
| `app.py` | **Step 5:** Full-stack FastAPI web dashboard for images, videos, YouTube streams, and webcam feeds. |
| `requirements.txt` | Project Python package dependencies. |
| `ppt.md` | 16-slide presentation deck with speaker notes and exact image asset mappings. |
| `expalin.md` | This comprehensive technical reference document. |
