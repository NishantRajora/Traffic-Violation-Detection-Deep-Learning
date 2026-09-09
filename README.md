# Helmet Detection & Traffic Violation Detection System

## 📌 Project Overview

The **Helmet Detection & Traffic Violation Detection System** is a Deep Learning and Computer Vision project designed to identify whether a two-wheeler rider is wearing a helmet.

The current implementation focuses on developing and evaluating a **YOLO11n image classification model** that classifies input images into two categories:

* `helmet`
* `no_helmet`

The project is developed as the initial Deep Learning component of a larger automated traffic violation detection system.

The long-term objective is to extend the system from simple helmet classification toward a complete traffic monitoring pipeline involving:

* Rider detection
* Helmet/no-helmet detection
* Vehicle identification
* Number plate detection
* Optical Character Recognition (OCR)
* Violation evidence generation

---

# 🎯 Project Objectives

The major objectives of this project are:

1. Develop a Deep Learning-based helmet detection system.
2. Train a lightweight YOLO11 classification model.
3. Classify images into helmet and no-helmet categories.
4. Evaluate the trained model using validation and test data.
5. Save the best-performing trained model.
6. Test the model on unseen images.
7. Study the limitations of image classification for real-world traffic scenes.
8. Establish a baseline for future rider-level traffic violation detection.
9. Extend the system toward number plate detection and OCR in future phases.

---

# 🧠 Problem Statement

Two-wheeler riders are required to wear helmets for road safety. However, manually monitoring helmet compliance across large traffic areas is difficult.

Traditional monitoring depends heavily on:

* CCTV operators
* Manual inspection
* Traffic police
* Random checking

This creates limitations in terms of:

* scalability
* response time
* continuous monitoring
* human effort

Computer Vision and Deep Learning can help automate the identification of helmet violations.

The proposed system investigates the use of YOLO-based Deep Learning to automatically classify helmet compliance from images.

---

# 💡 Proposed Solution

The current system follows this workflow:

```text
Input Image
     │
     ▼
Image Preprocessing
     │
     ▼
YOLO11n Classification Model
     │
     ▼
Feature Extraction
     │
     ▼
Classification
     │
     ├───────────────┐
     ▼               ▼
  HELMET        NO HELMET
     │               │
     ▼               ▼
Confidence      Confidence
 Score            Score
```

The current model performs **image classification**, meaning it predicts the class of the complete input image.

Ultralytics' classification models return the predicted class and confidence score rather than object bounding boxes.

---

# 🤖 Model Used

## YOLO11n Classification

The project uses:

```text
YOLO11n Classification
```

Model file:

```text
yolo11n-cls.pt
```

YOLO11 supports multiple computer vision tasks, including detection, segmentation, classification, pose estimation, and oriented bounding boxes. The `-cls` model variant is specifically intended for image classification.

The `n` represents the **nano** model variant, which is designed to provide a lightweight balance between computational requirements and model performance.

---

# 🔄 Transfer Learning

Instead of training the model completely from random initialization, the project starts from a pretrained YOLO11 classification model.

```text
Pretrained YOLO11n
       │
       ▼
Learned General Features
       │
       ▼
Helmet Dataset
       │
       ▼
Fine-Tuning
       │
       ▼
Helmet Classification Model
```

This approach allows the model to reuse previously learned visual features and adapt them to the helmet/no-helmet classification problem.

---

# 📂 Dataset

The dataset is organized into three major subsets:

```text
helmet_dataset/
│
├── train/
│   ├── helmet/
│   └── no helmet/
│
├── validation/
│   ├── helmet/
│   └── no helmet/
│
└── test/
    ├── helmet/
    └── no helmet/
```

## Classes

The project contains two classes:

```text
0 → helmet
1 → no_helmet
```

The source dataset uses the categories:

```text
helmet
no helmet
```

---

# 📊 Dataset Distribution

The current dataset contains approximately:

| Dataset Split | Images |
| ------------- | -----: |
| Training      |   1199 |
| Validation    |    200 |
| Testing       |    200 |
| Total         |  ~1599 |

The training set is used to learn the model parameters.

The validation set is used during model development to monitor performance.

The test set is reserved for evaluating the trained model on unseen data.

---

# 🧹 Dataset Validation

Before training, the dataset is checked for:

* Valid image files
* Correct folder structure
* Correct class names
* Missing files
* Corrupted files
* Unsupported image formats

Some files in the dataset were identified as invalid/corrupt despite having `.jpg` extensions and were excluded from the usable dataset.

This preprocessing step is important because invalid image files can cause errors during model training.

---

# 🔬 Exploratory Data Analysis

The project includes dataset inspection and analysis before model training.

The analysis focuses on:

* Dataset structure
* Number of images
* Class distribution
* Training/validation/test split
* Image availability
* Invalid files
* Sample images

The purpose of EDA is to understand the dataset before training the Deep Learning model.

---

# ⚙️ Training Configuration

The current training configuration is:

```text
Model          : YOLO11n Classification
Task           : Image Classification
Epochs         : 30
Image Size     : 640 × 640
Batch Size     : 16
Device         : NVIDIA GPU
Optimizer      : AdamW
Workers        : 4
```

The model is trained using the Ultralytics YOLO framework.

Ultralytics provides training, validation, prediction and export workflows through its Python API and command-line interface.

---

# 🏋️ Training Process

The training process can be represented as:

```text
Dataset
   │
   ▼
Data Loading
   │
   ▼
Image Preprocessing
   │
   ▼
YOLO11n Pretrained Model
   │
   ▼
Forward Pass
   │
   ▼
Prediction
   │
   ▼
Loss Calculation
   │
   ▼
Backpropagation
   │
   ▼
Optimizer Update
   │
   ▼
Validation
   │
   ▼
Next Epoch
```

This process is repeated for the configured number of epochs.

---

# 📉 Loss and Optimization

During training, the model calculates a loss value that represents the difference between its predictions and the expected class.

The optimizer then updates the model's parameters to reduce this loss.

The project uses:

```text
AdamW
```

as the optimizer.

The training loss decreased substantially during the training process, indicating that the model was learning the classification task.

---

# 📈 Model Evaluation

The model is evaluated using classification metrics.

The major metrics used include:

* Top-1 Accuracy
* Top-5 Accuracy
* Confidence Score

For image classification, Ultralytics exposes metrics such as `top1` and `top5` during validation.

---

# 🏆 Validation Results

The trained model achieved:

```text
Top-1 Accuracy : 100%
Top-5 Accuracy : 100%
```

These results were obtained during the project's validation/evaluation process.

However, high accuracy on a particular dataset should not automatically be interpreted as equivalent real-world performance.

Further testing on diverse traffic images and videos is required.

---

# 💾 Trained Model

The best trained model is stored as:

```text
saved_models/
└── helmet_model.pt
```

This file contains the trained YOLO11 classification model.

The model can be loaded using:

```python
from ultralytics import YOLO

model = YOLO("saved_models/helmet_model.pt")
```

---

# 🧪 Testing

The testing workflow is implemented in:

```text
testing.ipynb
```

and:

```text
test.py
```

The model can be tested using new images that were not used during training.

Example:

```python
from ultralytics import YOLO

model = YOLO("saved_models/helmet_model.pt")

results = model.predict("image.jpg")

for result in results:
    top1 = result.probs.top1
    confidence = result.probs.top1conf
    class_name = result.names[top1]

    print("Prediction:", class_name)
    print("Confidence:", confidence)
```

Ultralytics classification inference exposes the predicted class through `result.probs.top1` and its confidence through `result.probs.top1conf`.

---

# 🌐 External Image Testing

The project also includes testing on external images.

The basic workflow is:

```text
Internet Image
      │
      ▼
Download Image
      │
      ▼
Convert to RGB
      │
      ▼
YOLO11 Model
      │
      ▼
Prediction
      │
      ▼
Helmet / No Helmet
      │
      ▼
Confidence
```

This provides an additional way to check how the trained model behaves on images outside the original dataset.

---

# 🎥 Video Detection

## Current Status

The project explores video-based helmet violation detection, but the current trained model is a **classification model**.

Therefore, the current model should not be considered a complete multi-rider traffic video detector.

For example:

```text
Traffic Camera
       │
       ▼
Multiple Riders
       │
       ▼
YOLO11 Classification
       │
       ▼
Whole-image Classification
```

The model does not independently locate every rider.

---

# ⚠️ Why Classification Is Not Enough

Suppose a traffic image contains:

```text
Rider 1 → Helmet
Rider 2 → No Helmet
Rider 3 → Helmet
```

A classification model does not inherently produce:

```text
Rider 1 → Helmet
Rider 2 → No Helmet
Rider 3 → Helmet
```

with individual bounding boxes.

Instead, it predicts a class for the input image.

Object detection is the appropriate task when the system needs to identify **where individual objects are located** in an image.

---

# 🚧 Current Limitations

The current implementation does not yet provide:

* Individual rider detection
* Rider bounding boxes
* Multiple rider tracking
* Motorcycle detection
* Number plate detection
* Number plate OCR
* Vehicle identification
* Automatic challan generation
* Violation database
* Real-time traffic enforcement

Therefore, the current project should be described as:

> **YOLO11-based Helmet/No-Helmet Image Classification**

rather than claiming that it is already a complete automated traffic enforcement system.

---

# 🔬 Research Direction

The project can be extended into a complete multi-stage traffic violation detection system.

A possible future architecture is:

```text
             Traffic Video
                   │
                   ▼
          Rider Detection
                   │
                   ▼
          Helmet Detection
                   │
          ┌────────┴────────┐
          ▼                 ▼
       Helmet           No Helmet
                            │
                            ▼
                   Number Plate Detection
                            │
                            ▼
                       Plate Crop
                            │
                            ▼
                           OCR
                            │
                            ▼
                    Vehicle Number
                            │
                            ▼
                   Violation Record
```

---

# 🚀 Future Improvements

## 1. Object Detection

Replace or extend the current classification model with a YOLO object-detection model.

Potential classes:

```text
person
motorcycle
helmet
no_helmet
number_plate
```

This would allow the system to locate objects using bounding boxes.

---

## 2. Rider-Level Helmet Detection

Instead of classifying the entire image, detect individual riders.

Example:

```text
Rider 1 → Helmet
Rider 2 → No Helmet
Rider 3 → Helmet
```

This would make the system much more suitable for traffic surveillance.

---

## 3. Number Plate Detection

For riders classified as violating helmet rules, the system could detect the associated vehicle's number plate.

```text
No Helmet
    ↓
Vehicle
    ↓
Number Plate
```

---

## 4. OCR

After detecting the number plate, OCR can be applied to extract the vehicle registration number.

Example:

```text
Number Plate Image
        ↓
       OCR
        ↓
   DL01AB1234
```

---

## 5. Video Processing

The system can be extended to process:

* CCTV footage
* Traffic camera footage
* Uploaded videos
* Webcam streams

Ultralytics supports video and stream inputs for appropriate YOLO tasks, including object detection.

---

## 6. Real-Time Detection

Future development can focus on optimizing the model for:

* GPU inference
* Edge devices
* Real-time CCTV
* Low-latency processing
* Lightweight deployment

YOLO11 is designed to support different deployment environments, including NVIDIA GPU and edge-device scenarios.

---

# 📁 Project Structure

```text
Helmet-Detection/
│
├── helmet_dataset/
│   │
│   ├── train/
│   │   ├── helmet/
│   │   └── no helmet/
│   │
│   ├── validation/
│   │   ├── helmet/
│   │   └── no helmet/
│   │
│   └── test/
│       ├── helmet/
│       └── no helmet/
│
├── runs/
│   └── training results
│
├── saved_models/
│   └── helmet_model.pt
│
├── .gitignore
│
├── source.txt
│
├── test.py
│
├── testing.ipynb
│
├── training.ipynb
│
├── verify.py
│
├── yolo11n-cls.pt
│
├── yolo26n.pt
│
└── README.md
```

---

# 📄 File Description

| File                           | Description                             |
| ------------------------------ | --------------------------------------- |
| `helmet_dataset/`              | Helmet/no-helmet dataset                |
| `training.ipynb`               | Complete model training notebook        |
| `testing.ipynb`                | Model testing and prediction notebook   |
| `test.py`                      | Python testing script                   |
| `verify.py`                    | Dataset/model verification              |
| `saved_models/helmet_model.pt` | Trained model                           |
| `runs/`                        | Training outputs and experiment results |
| `yolo11n-cls.pt`               | Pretrained YOLO11 classification model  |
| `yolo26n.pt`                   | Additional YOLO model checkpoint        |
| `source.txt`                   | Source/reference information            |
| `.gitignore`                   | Git ignored files                       |
| `README.md`                    | Project documentation                   |

---

# 🛠️ Technologies

The project uses:

```text
Python
PyTorch
Ultralytics YOLO
YOLO11
OpenCV
NumPy
Pandas
Matplotlib
Jupyter Notebook
```

---

# 💻 Hardware Environment

The project was developed and trained using a CUDA-enabled NVIDIA GPU.

The training environment supports GPU acceleration through PyTorch and CUDA.

This allows the model training process to be significantly faster than CPU-only training.

---

# 📦 Installation

Clone the repository:

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd <PROJECT-FOLDER>
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install ultralytics
pip install torch torchvision
pip install opencv-python
pip install numpy pandas matplotlib
pip install jupyter
```

---

# ▶️ Running the Project

## Training

Open:

```text
training.ipynb
```

Run the notebook cells sequentially.

The trained model will be generated after training.

---

## Testing

Open:

```text
testing.ipynb
```

Load:

```text
saved_models/helmet_model.pt
```

Alternatively:

```bash
python test.py
```

---

# 🔄 Complete Current Workflow

```text
                DATASET
                   │
                   ▼
           Dataset Verification
                   │
                   ▼
          Data Preprocessing
                   │
                   ▼
             EDA / Analysis
                   │
                   ▼
        Pretrained YOLO11n-cls
                   │
                   ▼
           Transfer Learning
                   │
                   ▼
              Training
                   │
                   ▼
             Validation
                   │
                   ▼
              best.pt
                   │
                   ▼
        saved_models/helmet_model.pt
                   │
                   ▼
                Testing
                   │
                   ▼
          External Image Testing
                   │
                   ▼
       Helmet / No Helmet Prediction
```

---

# 📚 Research Significance

The project demonstrates how modern Deep Learning can be applied to an important road-safety problem.

The current implementation provides a foundation for investigating:

* Automated helmet compliance monitoring
* Computer Vision-based traffic surveillance
* Lightweight Deep Learning models
* Real-time inference
* Rider-level violation detection
* Number plate recognition
* Automated traffic violation systems

The project can subsequently move from **classification** toward **object detection + OCR**, creating a more complete traffic violation pipeline.

---

# 🔐 Responsible Use

A real-world traffic enforcement system would need to consider:

* Privacy
* Data protection
* False positives
* False negatives
* Camera quality
* Lighting conditions
* Occlusion
* Multiple riders
* Model bias
* Human verification
* Legal requirements

Therefore, model predictions should be carefully validated before being used for real-world enforcement decisions.

---

# 📌 Current Project Status

### Completed

* [x] Dataset organization
* [x] Dataset verification
* [x] Data preprocessing
* [x] Dataset analysis
* [x] YOLO11n classification model selection
* [x] Transfer learning
* [x] Model training
* [x] Model validation
* [x] Best model saving
* [x] Image testing
* [x] External image testing
* [x] Project documentation

### In Progress / Future

* [ ] Two-wheeler/rider object detection
* [ ] Individual rider helmet detection
* [ ] Number plate detection
* [ ] OCR
* [ ] Rider tracking
* [ ] Video-based violation detection
* [ ] Violation evidence generation
* [ ] Real-time deployment

---

# 📊 Final Result

The current system successfully demonstrates **helmet/no-helmet image classification using YOLO11n**.

The trained model achieves high validation accuracy on the available dataset and can classify new images with a confidence score.

However, the current implementation is a foundation rather than a complete traffic enforcement system.

The next major research step is to move from:

```text
IMAGE CLASSIFICATION
```

to:

```text
OBJECT DETECTION
        +
NUMBER PLATE DETECTION
        +
OCR
        +
VIDEO PROCESSING
```

This progression would transform the current helmet classification project into a more complete automated traffic violation detection system.

---

# 👨‍💻 Author

**Nishant Rajora**

B.Tech Computer Science and Engineering
Specialization: Data Science

---

# 📜 License

This project is intended primarily for educational and research purposes.

If this project is used or extended for commercial deployment, review the applicable licenses of the software, pretrained models, datasets, and other third-party components.

For YOLO11, refer to the current Ultralytics licensing and usage terms.

---

# ⭐ Acknowledgement

This project uses the **Ultralytics YOLO framework** for Deep Learning-based computer vision.

Official documentation:

https://docs.ultralytics.com/
