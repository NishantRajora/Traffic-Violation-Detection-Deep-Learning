# 🚦 Traffic Violation Detection Using YOLO

A deep-learning based **Traffic Violation Detection** project using the **YOLO (You Only Look Once)** framework for image classification.

The model is trained to classify traffic-related images into four categories:

* 🪖 **Helmet**
* ⚠️ **No Helmet**
* 🚛 **Overloading**
* 🌫️ **Blur**

The project uses a pretrained **YOLO classification model** and fine-tunes it on a custom traffic-violation dataset.

---

## 📌 Project Overview

Road safety violations such as riding without a helmet and vehicle overloading are common traffic violations.

This project uses computer vision and deep learning to automatically classify traffic images into predefined categories.

The overall workflow is:

```text
Traffic Images
      │
      ▼
Dataset Preparation
      │
      ▼
YOLO Classification Model
      │
      ▼
Transfer Learning / Fine-Tuning
      │
      ▼
Validation
      │
      ▼
Best Model (best.pt)
      │
      ▼
Testing
```

---

# 🧠 Technology Used

| Technology       | Purpose                 |
| ---------------- | ----------------------- |
| Python           | Programming language    |
| YOLO             | Image classification    |
| Ultralytics      | YOLO implementation     |
| PyTorch          | Deep learning framework |
| CUDA             | GPU acceleration        |
| NVIDIA RTX 4050  | Model training          |
| PIL              | Image processing        |
| Matplotlib       | Visualization           |
| Jupyter Notebook | Training environment    |

---

# 📂 Dataset Structure

The dataset is organized into three subsets:

```text
Traffic Violations Dataset/
│
├── train/
│   ├── blur/
│   ├── helmet/
│   ├── no_helmet/
│   └── overloading/
│
├── validation/
│   ├── blur/
│   ├── helmet/
│   ├── no_helmet/
│   └── overloading/
│
└── test/
    ├── blur/
    ├── helmet/
    ├── no_helmet/
    └── overloading/
```

Each class is represented by a separate folder.

### Classes

```text
blur
helmet
no_helmet
overloading
```

The model learns to classify an input image into one of these four categories.

---

# 🔍 Classification vs Object Detection

This project currently uses **YOLO image classification**.

The model predicts the class of the complete image.

For example:

```text
Input Image
     │
     ▼
YOLO Classification Model
     │
     ▼
no_helmet
94.32%
```

It does **not** generate bounding boxes around individual objects.

For bounding-box based detection, a YOLO object-detection dataset with corresponding annotation files would be required.

---

# 🤖 YOLO Model

The project uses a pretrained YOLO classification model:

```python
YOLO("yolo11n-cls.pt")
```

The `-cls` suffix indicates that this is a **classification model**.

The pretrained model is fine-tuned using the custom traffic violation dataset.

---

# 🔄 Transfer Learning

Instead of training the neural network completely from scratch, a pretrained YOLO model is used.

```text
Pretrained YOLO
       │
       │
       ▼
Traffic Violation Dataset
       │
       ▼
Fine-Tuning
       │
       ▼
Traffic Violation Model
```

The pretrained model already contains useful visual features.

Fine-tuning allows it to adapt those features to the four traffic-related classes.

---

# ⚙️ Hardware Configuration

Training was performed using an NVIDIA GPU.

### GPU

```text
NVIDIA GeForce RTX 4050 Laptop GPU
```

### CUDA

The training environment was configured with CUDA support.

The following code is used to verify GPU availability:

```python
import torch

print("PyTorch Version :", torch.__version__)
print("CUDA Version    :", torch.version.cuda)
print("CUDA Available  :", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU :", torch.cuda.get_device_name(0))
else:
    print("WARNING: GPU not detected. Training will use CPU.")
```

Expected output:

```text
PyTorch Version : ...
CUDA Version    : ...
CUDA Available  : True
GPU             : NVIDIA GeForce RTX 4050 Laptop GPU
```

---

# 📦 Installation

Install the required packages:

```bash
pip install ultralytics torch torchvision
```

Additional packages used in the project:

```bash
pip install pillow matplotlib
```

---

# 🏋️ Training

The model is trained using the Ultralytics YOLO framework.

Basic model initialization:

```python
from ultralytics import YOLO

model = YOLO("yolo11n-cls.pt")
```

The dataset is then supplied to the training process.

Example training configuration:

```python
results = model.train(

    data="Traffic Violations Dataset",

    epochs=50,

    batch=32,

    imgsz=224,

    device="0",

    workers=4,

    project="Traffic_Violation_Runs",

    name="helmet_classification",

    save=True,

    val=True,

    patience=10,

    cache=False,

    verbose=True
)
```

---

# ⚙️ Training Parameters

| Parameter  | Value | Description                          |
| ---------- | ----: | ------------------------------------ |
| `epochs`   |    50 | Number of training epochs            |
| `batch`    |    32 | Number of images processed per batch |
| `imgsz`    |   224 | Input image size                     |
| `device`   |   `0` | NVIDIA GPU                           |
| `workers`  |     4 | Data-loading workers                 |
| `val`      |  True | Perform validation                   |
| `save`     |  True | Save model checkpoints               |
| `patience` |    10 | Early stopping patience              |
| `cache`    | False | Dataset caching disabled             |

These values can be adjusted depending on available hardware and dataset size.

---

# 📚 Important Training Concepts

## Epoch

An epoch represents one complete pass through the training dataset.

For example:

```text
Epoch 1
Epoch 2
Epoch 3
...
Epoch 50
```

---

## Batch Size

The batch size specifies how many images are processed before the model updates its weights.

For example:

```text
batch = 32
```

means approximately 32 images are processed at a time.

---

## Image Size

```text
imgsz = 224
```

means the images are processed at approximately:

```text
224 × 224
```

Higher image sizes can preserve more visual detail but require more computational resources.

---

## Workers

```text
workers = 4
```

controls the number of parallel processes used for loading and preparing training data.

Workers primarily help with data loading and preprocessing.

---

# 📈 Accuracy During Training

The model's performance is monitored after each epoch.

The project tracks:

```text
Top-1 Accuracy
Top-5 Accuracy
```

For example:

```text
Epoch 1
Top-1 Accuracy: 72.00%

Epoch 2
Top-1 Accuracy: 79.00%

Epoch 3
Top-1 Accuracy: 84.00%

...
```

### Top-1 Accuracy

Top-1 accuracy measures whether the model's highest-confidence prediction matches the actual class.

For this project, **Top-1 accuracy is the primary accuracy metric**.

---

# 🧪 Validation Dataset

The validation dataset is used during training to monitor how well the model generalizes to images that are not directly used for updating the model's weights.

```text
Training Dataset
      ↓
Model learns
      ↓
Validation Dataset
      ↓
Performance monitoring
```

Validation accuracy can be monitored after every epoch.

---

# 🧪 Testing Dataset

The test dataset is kept separate from training.

After training is complete, the best model is evaluated using the test dataset.

```text
Training
    ↓
Validation
    ↓
Best Model
    ↓
Test Dataset
    ↓
Final Performance
```

The test accuracy provides an estimate of how the trained model performs on unseen test images.

---

# 🏆 Best Model

During training, YOLO saves model checkpoints.

The important file is:

```text
best.pt
```

A typical training output directory is:

```text
Traffic_Violation_Runs/
└── helmet_classification/
    └── weights/
        ├── best.pt
        └── last.pt
```

### `best.pt`

Contains the checkpoint selected as the best-performing model during training/validation.

### `last.pt`

Contains the checkpoint from the final training epoch.

For final inference, `best.pt` is generally used.

---

# 🔬 Model Evaluation

After training, the best model can be loaded:

```python
from ultralytics import YOLO

model = YOLO(
    "Traffic_Violation_Runs/helmet_classification/weights/best.pt"
)
```

The model can then be evaluated on the test dataset:

```python
metrics = model.val(
    data="Traffic Violations Dataset",
    split="test",
    imgsz=224,
    batch=32,
    device="0"
)
```

The final Top-1 accuracy can be displayed using:

```python
print(
    f"Top-1 Accuracy: {metrics.top1 * 100:.2f}%"
)
```

---

# 📊 Training vs Testing Accuracy

These two measurements have different purposes.

### During training

The model's validation performance is monitored after each epoch:

```text
Epoch 1 → Validation Accuracy
Epoch 2 → Validation Accuracy
Epoch 3 → Validation Accuracy
...
```

### After training

The final trained model is evaluated on the separate test dataset:

```text
Best Model
    ↓
Test Dataset
    ↓
Final Test Accuracy
```

The test dataset should not be used to update the model's weights.

---

# 📁 Expected Project Structure

After training, the project can look like:

```text
Traffic_Violation_Project/
│
├── Traffic Violations Dataset/
│   ├── train/
│   │   ├── blur/
│   │   ├── helmet/
│   │   ├── no_helmet/
│   │   └── overloading/
│   │
│   ├── validation/
│   │   ├── blur/
│   │   ├── helmet/
│   │   ├── no_helmet/
│   │   └── overloading/
│   │
│   └── test/
│       ├── blur/
│       ├── helmet/
│       ├── no_helmet/
│       └── overloading/
│
├── Traffic_Violation_Runs/
│   └── helmet_classification/
│       └── weights/
│           ├── best.pt
│           └── last.pt
│
├── saved_models/
│   └── helmet_model.pt
│
└── training.ipynb
```

---

# 🚀 Workflow Summary

```text
                  DATASET
                     │
                     ▼
          ┌─────────────────────┐
          │ Train / Validation  │
          │       / Test        │
          └──────────┬──────────┘
                     │
                     ▼
             Pretrained YOLO
              yolo11n-cls.pt
                     │
                     ▼
                Fine-Tuning
                     │
                     ▼
               Epoch Training
                     │
                     ▼
             Validation Accuracy
                     │
                     ▼
                  best.pt
                     │
                     ▼
              Test Evaluation
                     │
                     ▼
              Final Accuracy
```

---

# ⚠️ Current Project Limitation

The current model performs **image-level classification**.

For example:

```text
Image
  ↓
YOLO
  ↓
NO_HELMET — 94%
```

It does not identify the exact location of the rider or helmet.

For a complete traffic violation detection system, a future version can use **YOLO object detection** with bounding boxes to detect individual riders, helmets, motorcycles, and other objects.

---

# 👨‍💻 Author

**Nishant Rajora**

B.Tech Computer Science & Engineering
Specialization: Data Science

The NorthCap University, Gurugram

---

# 📌 Project Status

**Current Stage:**

* ✅ Dataset preparation
* ✅ YOLO classification model
* ✅ GPU configuration
* ✅ Model training
* ✅ Validation
* ✅ Best model generation
* ✅ Test evaluation

**Next Stage:**

* Web-based image testing
* FastAPI backend
* Drag-and-drop image interface
* Model inference API
* Traffic violation detection interface
