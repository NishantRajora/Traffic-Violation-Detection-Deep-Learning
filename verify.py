import os

# Prevent OpenMP runtime conflict crash on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import torch
import ultralytics
import yaml
from ultralytics import YOLO
#+++++


print(f"PyTorch Version: {torch.__version__}")
print(f"PyTorch CUDA Version: {torch.version.cuda}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"OpenCV Version: {cv2.__version__}")
print(f"Ultralytics Version: {ultralytics.__version__}")

# Verify NVIDIA RTX 4050 GPU
if torch.cuda.is_available():
    print(f"CUDA is active on GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Count: {torch.cuda.device_count()}")
else:
    print("WARNING: GPU not detected. PyTorch will run on CPU.")


