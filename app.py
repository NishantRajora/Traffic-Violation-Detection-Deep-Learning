import os
# Prevent OpenMP library duplicate conflicts on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import io
import base64
import shutil
from pathlib import Path
from typing import Optional
from PIL import Image
import requests
import cv2
import numpy as np
import torch
from ultralytics import YOLO

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Initialize FastAPI App
app = FastAPI(
    title="Traffic Violation Detection System",
    description="Real-Time Object Detection & Bounding Box Localization with PyTorch & YOLO11",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# Hardware & Model Initialization
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "Traffic Violations Dataset"

# Hardware Selection
if torch.cuda.is_available():
    DEVICE = 0
    DEVICE_NAME = torch.cuda.get_device_name(0)
else:
    DEVICE = "cpu"
    DEVICE_NAME = "CPU"

print(f"Inference Device: {DEVICE_NAME}")

# Locate Detection Model (Plots Bounding Boxes)
CANDIDATE_DETECTION_PATHS = [
    BASE_DIR / "saved_models" / "helmet_detection_model.pt",
    BASE_DIR / "runs" / "detect" / "Traffic_Detection_Runs" / "helmet_detection-5" / "weights" / "best.pt",
    BASE_DIR / "runs" / "detect" / "Traffic_Detection_Runs" / "helmet_detection" / "weights" / "best.pt",
]

DETECTION_MODEL_PATH = None
for candidate in CANDIDATE_DETECTION_PATHS:
    if candidate.exists():
        DETECTION_MODEL_PATH = candidate
        break

detection_model = None
if DETECTION_MODEL_PATH:
    print(f"Loading Object Detection Model from: {DETECTION_MODEL_PATH}")
    detection_model = YOLO(str(DETECTION_MODEL_PATH))
    # Warm up model
    dummy = Image.new("RGB", (320, 320), color=(128, 128, 128))
    detection_model.predict(dummy, device=DEVICE, conf=0.10, verbose=False)
    print(f"Detection Model Loaded Successfully! Task: {detection_model.task} | Classes: {detection_model.names}")
else:
    print("CRITICAL: Object Detection Model weights not found in expected paths.")

# Locate Classification Model (Whole Image Classification)
CANDIDATE_CLASSIFICATION_PATHS = [
    BASE_DIR / "saved_models" / "traffic_violation_model.pt",
    BASE_DIR / "saved_models" / "helmet_model.pt",
    BASE_DIR / "runs" / "classify" / "Traffic_Violation_Runs" / "helmet_classification" / "weights" / "best.pt",
]

CLASSIFICATION_MODEL_PATH = None
for candidate in CANDIDATE_CLASSIFICATION_PATHS:
    if candidate.exists():
        CLASSIFICATION_MODEL_PATH = candidate
        break

classification_model = None
if CLASSIFICATION_MODEL_PATH:
    print(f"Loading Classification Model from: {CLASSIFICATION_MODEL_PATH}")
    classification_model = YOLO(str(CLASSIFICATION_MODEL_PATH))
    print(f"Classification Model Loaded! Task: {classification_model.task} | Classes: {classification_model.names}")

# Class styling and configuration
CLASS_META = {
    "no_helmet": {
        "label": "No Helmet",
        "type": "violation",
        "color": "#ef4444",
        "icon": "🚨"
    },
    "overloading": {
        "label": "Overloading",
        "type": "violation",
        "color": "#dc2626",
        "icon": "🚛"
    },
    "helmet": {
        "label": "Helmet",
        "type": "compliant",
        "color": "#10b981",
        "icon": "🪖"
    },
    "blur": {
        "label": "Blur",
        "type": "warning",
        "color": "#f59e0b",
        "icon": "🌫️"
    }
}

# Curated High-Performing Sample Images
SAMPLE_IMAGES = [
    {
        "id": "sample_helmet",
        "class": "helmet",
        "label": "🪖 Helmet (Safe)",
        "file_path": "Traffic Violations Dataset/test/helmet/Test helmmet (19).jpg"
    },
    {
        "id": "sample_no_helmet",
        "class": "no_helmet",
        "label": "🚨 No Helmet (Violation)",
        "file_path": "Traffic Violations Dataset/test/no_helmet/test-nohelmet (3).jpg"
    },
    {
        "id": "sample_multi",
        "class": "no_helmet",
        "label": "🚨 Multi-Riders Without Helmet",
        "file_path": "Traffic Violations Dataset/test/no_helmet/test-nohelmet (4).jpg"
    },
    {
        "id": "sample_overloading",
        "class": "overloading",
        "label": "🚛 Overloading (Violation)",
        "file_path": "Traffic Violations Dataset/test/overloading/test-nohelmet (28).jpg"
    }
]

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def image_to_base64(img: Image.Image, format="JPEG", quality=90) -> str:
    buffered = io.BytesIO()
    img.save(buffered, format=format, quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")

def run_object_detection(image: Image.Image, conf_threshold: float = 0.15) -> dict:
    """Runs YOLO object detection, renders bounding boxes, and returns summary."""
    if detection_model is None:
        raise HTTPException(
            status_code=500,
            detail="Object Detection model is not available. Please ensure saved_models/helmet_detection_model.pt exists."
        )

    start_time = time.perf_counter()
    if image.mode != "RGB":
        image = image.convert("RGB")

    results = detection_model.predict(source=image, device=DEVICE, conf=conf_threshold, verbose=False)
    inference_ms = round((time.perf_counter() - start_time) * 1000, 2)

    result = results[0]

    # Render bounding boxes onto the image with clean line width and font size
    annotated_bgr = result.plot(line_width=3, font_size=1)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    annotated_pil = Image.fromarray(annotated_rgb)
    annotated_b64 = image_to_base64(annotated_pil)

    # Extract detected boxes
    boxes_data = []
    violation_count = 0
    compliant_count = 0

    if result.boxes is not None and len(result.boxes) > 0:
        for b in result.boxes:
            cls_id = int(b.cls[0])
            conf = float(b.conf[0])
            xyxy = [round(float(coord), 1) for coord in b.xyxy[0].tolist()]
            cls_name = detection_model.names.get(cls_id, f"Class {cls_id}")

            meta = CLASS_META.get(cls_name, {
                "label": cls_name.replace("_", " ").title(),
                "type": "info",
                "color": "#3b82f6",
                "icon": "ℹ️"
            })

            if meta["type"] == "violation":
                violation_count += 1
            elif meta["type"] == "compliant":
                compliant_count += 1

            boxes_data.append({
                "class": cls_name,
                "label": meta["label"],
                "confidence_pct": round(conf * 100, 1),
                "bbox": xyxy,
                "color": meta["color"],
                "icon": meta["icon"],
                "type": meta["type"]
            })

    total_detections = len(boxes_data)

    if violation_count > 0:
        status_text = "TRAFFIC VIOLATION DETECTED"
        status_sub = f"{violation_count} violation box(es) detected"
        status_color = "#ef4444"
        status_bg = "rgba(239, 68, 68, 0.16)"
        status_icon = "🚨"
    elif compliant_count > 0:
        status_text = "COMPLIANT / SAFE"
        status_sub = f"All {compliant_count} detected rider(s) wearing helmets"
        status_color = "#10b981"
        status_bg = "rgba(16, 185, 129, 0.16)"
        status_icon = "✅"
    else:
        status_text = "NO OBJECTS DETECTED"
        status_sub = f"No targets found above {int(conf_threshold*100)}% confidence threshold. Try lowering slider."
        status_color = "#94a3b8"
        status_bg = "rgba(148, 163, 184, 0.12)"
        status_icon = "🔍"

    return {
        "mode": "detect",
        "status": status_text,
        "status_sub": status_sub,
        "status_color": status_color,
        "status_bg": status_bg,
        "status_icon": status_icon,
        "total_detections": total_detections,
        "violation_count": violation_count,
        "compliant_count": compliant_count,
        "boxes": boxes_data,
        "annotated_image": annotated_b64,
        "inference_ms": inference_ms,
        "device": DEVICE_NAME,
        "conf_used": round(conf_threshold, 2),
        "image_size": f"{image.width} × {image.height}"
    }

def run_image_classification(image: Image.Image) -> dict:
    """Runs whole-image classification."""
    if classification_model is None:
        raise HTTPException(status_code=500, detail="Classification model is not loaded.")

    start_time = time.perf_counter()
    if image.mode != "RGB":
        image = image.convert("RGB")

    results = classification_model.predict(source=image, device=DEVICE, verbose=False)
    inference_ms = round((time.perf_counter() - start_time) * 1000, 2)

    result = results[0]
    top1_idx = int(result.probs.top1)
    confidence = float(result.probs.top1conf)
    predicted_class = classification_model.names[top1_idx]

    probabilities = {}
    for idx, class_name in classification_model.names.items():
        probabilities[class_name] = round(float(result.probs.data[idx]) * 100, 2)
    sorted_probs = dict(sorted(probabilities.items(), key=lambda item: item[1], reverse=True))

    meta = CLASS_META.get(predicted_class, {
        "label": predicted_class,
        "type": "info",
        "color": "#3b82f6",
        "icon": "ℹ️"
    })

    if meta["type"] == "violation":
        status_text = "TRAFFIC VIOLATION DETECTED"
        status_color = "#ef4444"
        status_bg = "rgba(239, 68, 68, 0.16)"
        status_icon = "🚨"
    elif meta["type"] == "compliant":
        status_text = "COMPLIANT / SAFE"
        status_color = "#10b981"
        status_bg = "rgba(16, 185, 129, 0.16)"
        status_icon = "✅"
    else:
        status_text = "UNREADABLE / LOW QUALITY"
        status_color = "#f59e0b"
        status_bg = "rgba(245, 158, 11, 0.16)"
        status_icon = "⚠️"

    return {
        "mode": "classify",
        "class": predicted_class,
        "label": meta["label"],
        "confidence_pct": round(confidence * 100, 1),
        "status": status_text,
        "status_sub": f"Whole image classified as {meta['label']}",
        "status_color": status_color,
        "status_bg": status_bg,
        "status_icon": status_icon,
        "all_probabilities": sorted_probs,
        "inference_ms": inference_ms,
        "device": DEVICE_NAME,
        "image_size": f"{image.width} × {image.height}"
    }

# ---------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------
@app.get("/api/info")
def get_system_info():
    return {
        "detection_model": DETECTION_MODEL_PATH.name if DETECTION_MODEL_PATH else None,
        "detection_classes": detection_model.names if detection_model else {},
        "classification_model": CLASSIFICATION_MODEL_PATH.name if CLASSIFICATION_MODEL_PATH else None,
        "classification_classes": classification_model.names if classification_model else {},
        "device": DEVICE_NAME,
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available()
    }

@app.post("/api/predict")
async def predict_image(
    file: UploadFile = File(...),
    mode: str = Form("detect"),
    conf: float = Form(0.15)
):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        if mode == "detect" and detection_model is not None:
            return JSONResponse(content=run_object_detection(image, conf_threshold=conf))
        else:
            return JSONResponse(content=run_image_classification(image))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference failed: {str(e)}")

@app.post("/api/predict-url")
def predict_image_url(
    url: str = Form(...),
    mode: str = Form("detect"),
    conf: float = Form(0.15)
):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        res = requests.get(url.strip(), headers=headers, timeout=15)
        res.raise_for_status()
        image = Image.open(io.BytesIO(res.content))
        if mode == "detect" and detection_model is not None:
            return JSONResponse(content=run_object_detection(image, conf_threshold=conf))
        else:
            return JSONResponse(content=run_image_classification(image))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"URL inference failed: {str(e)}")

@app.get("/api/samples")
def get_sample_images():
    """Returns curated test sample images with high detection performance."""
    results = []
    for s in SAMPLE_IMAGES:
        p = BASE_DIR / s["file_path"]
        if p.exists():
            results.append({
                "id": s["id"],
                "class": s["class"],
                "label": s["label"],
                "path": f"/api/sample-file?id={s['id']}"
            })
    return results

@app.get("/api/sample-file")
def get_sample_file(id: str):
    for s in SAMPLE_IMAGES:
        if s["id"] == id:
            p = BASE_DIR / s["file_path"]
            if p.exists():
                return FileResponse(p, media_type="image/jpeg")
    raise HTTPException(status_code=404, detail="Sample image not found")

# ---------------------------------------------------------
# Web Frontend
# ---------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def serve_homepage():
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Traffic Violation Detection | Bounding Box Object Detection</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-body: #0a0e17;
            --bg-card: #111827;
            --bg-card-hover: #182235;
            --border: #1f293d;
            --border-hover: #374151;
            --text-main: #f9fafb;
            --text-muted: #94a3b8;
            --primary: #3b82f6;
            --primary-glow: rgba(59, 130, 246, 0.35);
            --danger: #ef4444;
            --success: #10b981;
            --warning: #f59e0b;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
        }

        body {
            background-color: var(--bg-body);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            background-image: 
                radial-gradient(at 10% 15%, rgba(59, 130, 246, 0.09) 0px, transparent 50%),
                radial-gradient(at 90% 85%, rgba(239, 68, 68, 0.09) 0px, transparent 50%);
        }

        /* Header */
        header {
            border-bottom: 1px solid var(--border);
            backdrop-filter: blur(12px);
            background: rgba(17, 24, 39, 0.75);
            position: sticky;
            top: 0;
            z-index: 50;
            padding: 0.9rem 2rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 0.85rem;
        }

        .brand-icon {
            font-size: 1.8rem;
            background: rgba(59, 130, 246, 0.12);
            border: 1px solid var(--border);
            width: 44px;
            height: 44px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 12px;
        }

        .brand-title {
            font-size: 1.25rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            background: linear-gradient(135deg, #ffffff 40%, #94a3b8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .brand-sub {
            font-size: 0.75rem;
            color: var(--text-muted);
            font-weight: 500;
        }

        .header-actions {
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        .badge-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.35rem 0.8rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            background: rgba(16, 185, 129, 0.12);
            color: #10b981;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }

        .badge-dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: #10b981;
            box-shadow: 0 0 8px #10b981;
        }

        /* Container */
        .container {
            max-width: 1260px;
            margin: 1.75rem auto;
            padding: 0 1.5rem;
            width: 100%;
            flex: 1;
        }

        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.75rem;
            align-items: start;
        }

        @media (max-width: 960px) {
            .grid {
                grid-template-columns: 1fr;
            }
        }

        /* Panel Styles */
        .panel {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 1.6rem;
            box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5);
            position: relative;
            overflow: hidden;
        }

        .panel-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.25rem;
        }

        .panel-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: var(--text-main);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        /* Mode Switcher Tabs */
        .mode-toggle {
            display: flex;
            background: #0b1120;
            padding: 4px;
            border-radius: 12px;
            border: 1px solid var(--border);
            margin-bottom: 1.1rem;
            gap: 4px;
        }

        .mode-btn {
            flex: 1;
            background: transparent;
            border: none;
            color: var(--text-muted);
            padding: 0.55rem 0.8rem;
            border-radius: 8px;
            font-size: 0.825rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.45rem;
        }

        .mode-btn.active {
            background: var(--primary);
            color: #ffffff;
            box-shadow: 0 2px 10px rgba(59, 130, 246, 0.35);
        }

        /* Confidence Slider */
        .slider-box {
            background: #0c1222;
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 0.75rem 1rem;
            margin-bottom: 1.1rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
        }

        .slider-label {
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--text-muted);
            white-space: nowrap;
        }

        .slider-control {
            display: flex;
            align-items: center;
            gap: 0.85rem;
            flex: 1;
        }

        input[type="range"] {
            flex: 1;
            accent-color: var(--primary);
            cursor: pointer;
        }

        .slider-val {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            font-weight: 700;
            color: var(--primary);
            min-width: 38px;
        }

        /* Drag and Drop Zone */
        .dropzone {
            border: 2px dashed var(--border);
            border-radius: 16px;
            padding: 2.2rem 1.5rem;
            text-align: center;
            background: rgba(15, 23, 42, 0.45);
            cursor: pointer;
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
        }

        .dropzone:hover, .dropzone.dragover {
            border-color: var(--primary);
            background: rgba(59, 130, 246, 0.08);
            box-shadow: 0 0 25px var(--primary-glow);
            transform: scale(1.008);
        }

        .dropzone-icon {
            font-size: 2.8rem;
            margin-bottom: 0.65rem;
            display: inline-block;
            transition: transform 0.2s;
        }

        .dropzone:hover .dropzone-icon {
            transform: translateY(-4px);
        }

        .dropzone-title {
            font-size: 1rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
            color: #f3f4f6;
        }

        .dropzone-desc {
            font-size: 0.8rem;
            color: var(--text-muted);
            margin-bottom: 0.9rem;
        }

        .browse-btn {
            background: #1f2937;
            color: #f9fafb;
            border: 1px solid var(--border-hover);
            padding: 0.45rem 1.1rem;
            border-radius: 8px;
            font-size: 0.8rem;
            font-weight: 600;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            transition: all 0.2s;
        }

        .browse-btn:hover {
            background: #374151;
            border-color: #4b5563;
        }

        #fileInput {
            display: none;
        }

        /* URL Input Section */
        .url-box {
            margin-top: 1rem;
            display: flex;
            gap: 0.5rem;
        }

        .url-input {
            flex: 1;
            background: #0c1222;
            border: 1px solid var(--border);
            color: #f3f4f6;
            padding: 0.6rem 0.9rem;
            border-radius: 10px;
            font-size: 0.825rem;
            outline: none;
            transition: border-color 0.2s;
        }

        .url-input:focus {
            border-color: var(--primary);
        }

        .url-btn {
            background: var(--primary);
            color: #fff;
            border: none;
            padding: 0.6rem 1.1rem;
            border-radius: 10px;
            font-size: 0.825rem;
            font-weight: 600;
            cursor: pointer;
            transition: opacity 0.2s;
            white-space: nowrap;
        }

        .url-btn:hover {
            opacity: 0.9;
        }

        /* Test Samples Grid */
        .samples-section {
            margin-top: 1.25rem;
        }

        .samples-label {
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-bottom: 0.65rem;
        }

        .samples-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 0.6rem;
        }

        .sample-card {
            background: #0c1222;
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 0.5rem;
            text-align: center;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 0.3rem;
        }

        .sample-card:hover {
            border-color: var(--primary);
            background: #1e293b;
            transform: translateY(-2px);
        }

        .sample-thumb {
            width: 100%;
            height: 52px;
            border-radius: 6px;
            object-fit: cover;
            background: #1e293b;
        }

        .sample-title {
            font-size: 0.7rem;
            font-weight: 600;
            color: #e2e8f0;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            width: 100%;
        }

        /* Right Panel: Preview and Detection Display */
        .view-toggle {
            display: flex;
            gap: 4px;
            background: #0c1222;
            padding: 3px;
            border-radius: 8px;
            border: 1px solid var(--border);
        }

        .view-btn {
            background: transparent;
            border: none;
            color: var(--text-muted);
            padding: 0.35rem 0.75rem;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 600;
            cursor: pointer;
        }

        .view-btn.active {
            background: #1f293d;
            color: #fff;
        }

        .preview-wrapper {
            position: relative;
            width: 100%;
            height: 320px;
            border-radius: 14px;
            overflow: hidden;
            background: #070a10;
            border: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 1.1rem;
        }

        .preview-img {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
            display: none;
        }

        .empty-placeholder {
            text-align: center;
            color: var(--text-muted);
            padding: 1.5rem;
        }

        .empty-placeholder span {
            font-size: 2.4rem;
            display: block;
            margin-bottom: 0.5rem;
            opacity: 0.5;
        }

        /* Result Alert Banner */
        .alert-banner {
            padding: 1rem 1.25rem;
            border-radius: 14px;
            display: none;
            align-items: center;
            gap: 0.9rem;
            margin-bottom: 1.1rem;
            border: 1px solid transparent;
            transition: all 0.3s;
        }

        .alert-icon {
            font-size: 2.2rem;
            line-height: 1;
        }

        .alert-details {
            flex: 1;
        }

        .alert-status {
            font-size: 0.75rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.15rem;
        }

        .alert-sub {
            font-size: 0.95rem;
            font-weight: 600;
            color: #f1f5f9;
        }

        /* Metrics Row */
        .metrics-row {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0.65rem;
            margin-bottom: 1.1rem;
        }

        .metric-card {
            background: #0c1222;
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 0.65rem 0.8rem;
        }

        .metric-label {
            font-size: 0.68rem;
            color: var(--text-muted);
            text-transform: uppercase;
            font-weight: 600;
            margin-bottom: 0.2rem;
        }

        .metric-value {
            font-size: 1.1rem;
            font-weight: 800;
            font-family: 'JetBrains Mono', monospace;
            color: #f9fafb;
        }

        /* Bounding Boxes List */
        .boxes-section-title {
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-bottom: 0.65rem;
            display: flex;
            justify-content: space-between;
        }

        .boxes-list {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
            max-height: 220px;
            overflow-y: auto;
            padding-right: 4px;
        }

        .box-item {
            background: #0c1222;
            border: 1px solid var(--border);
            border-left-width: 4px;
            border-radius: 8px;
            padding: 0.55rem 0.8rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .box-info {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.825rem;
            font-weight: 700;
        }

        .box-coords {
            font-size: 0.7rem;
            font-family: 'JetBrains Mono', monospace;
            color: var(--text-muted);
        }

        .box-conf {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
            font-weight: 700;
            padding: 0.2rem 0.5rem;
            border-radius: 6px;
            background: rgba(255, 255, 255, 0.05);
        }

        /* Probability Bars (For Classification Mode) */
        .prob-item {
            margin-bottom: 0.6rem;
        }

        .prob-header {
            display: flex;
            justify-content: space-between;
            font-size: 0.8rem;
            font-weight: 600;
            margin-bottom: 0.2rem;
        }

        .prob-track {
            height: 7px;
            background: #1f293d;
            border-radius: 9999px;
            overflow: hidden;
        }

        .prob-fill {
            height: 100%;
            border-radius: 9999px;
            width: 0%;
            transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
        }

        /* Loading Spinner Overlay */
        .loading-overlay {
            position: absolute;
            inset: 0;
            background: rgba(10, 14, 23, 0.85);
            backdrop-filter: blur(4px);
            display: none;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            z-index: 20;
            border-radius: 20px;
        }

        .spinner {
            width: 42px;
            height: 42px;
            border: 3px solid rgba(59, 130, 246, 0.2);
            border-top-color: var(--primary);
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
            margin-bottom: 0.75rem;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        /* Footer */
        footer {
            border-top: 1px solid var(--border);
            padding: 1.25rem 2rem;
            text-align: center;
            font-size: 0.8rem;
            color: var(--text-muted);
            background: rgba(17, 24, 39, 0.5);
        }

        footer strong {
            color: var(--text-main);
        }
    </style>
</head>
<body>

    <!-- Header -->
    <header>
        <div class="brand">
            <div class="brand-icon">🚦</div>
            <div>
                <div class="brand-title">Traffic Violation Detection</div>
                <div class="brand-sub">YOLO11 Object Detection & Bounding Box Localization</div>
            </div>
        </div>
        <div class="header-actions">
            <div class="badge-chip">
                <div class="badge-dot"></div>
                <span id="deviceBadge">NVIDIA RTX 4050 GPU Active</span>
            </div>
        </div>
    </header>

    <!-- Main Container -->
    <main class="container">
        <div class="grid">
            
            <!-- Left Column: Input / Drag-and-Drop -->
            <div class="panel">
                <div class="panel-header">
                    <div class="panel-title">
                        <span>📤</span> Input Traffic Image
                    </div>
                    <span style="font-size: 0.75rem; color: var(--text-muted);">PNG, JPG, JPEG</span>
                </div>

                <!-- Mode Switcher -->
                <div class="mode-toggle">
                    <button class="mode-btn active" id="btnModeDetect" onclick="setMode('detect')">
                        <span>🎯</span> Object Detection (Bounding Boxes)
                    </button>
                    <button class="mode-btn" id="btnModeClassify" onclick="setMode('classify')">
                        <span>🖼️</span> Classification
                    </button>
                </div>

                <!-- Confidence Slider (For Object Detection) -->
                <div class="slider-box" id="sliderContainer">
                    <span class="slider-label">Detection Confidence:</span>
                    <div class="slider-control">
                        <input type="range" id="confSlider" min="5" max="80" value="15" oninput="updateConf(this.value)">
                        <span class="slider-val" id="confVal">15%</span>
                    </div>
                </div>

                <!-- Dropzone -->
                <div class="dropzone" id="dropzone">
                    <span class="dropzone-icon">📷</span>
                    <div class="dropzone-title">Drag & Drop traffic image here</div>
                    <div class="dropzone-desc">or paste an image from clipboard (Ctrl + V)</div>
                    <button class="browse-btn" type="button" onclick="document.getElementById('fileInput').click()">
                        📁 Browse Computer
                    </button>
                    <input type="file" id="fileInput" accept="image/*">
                </div>

                <!-- URL Input -->
                <div class="url-box">
                    <input type="url" id="urlInput" class="url-input" placeholder="Or paste image URL (https://...)">
                    <button class="url-btn" onclick="analyzeUrl()">Analyze URL</button>
                </div>

                <!-- Quick Samples -->
                <div class="samples-section">
                    <div class="samples-label">⚡ One-Click Test Samples (with Bounding Boxes):</div>
                    <div class="samples-grid" id="samplesGrid">
                        <div class="sample-card" style="opacity: 0.5;">Loading samples...</div>
                    </div>
                </div>
            </div>

            <!-- Right Column: Results & Analytics -->
            <div class="panel" style="position: relative;">
                <!-- Loading Overlay -->
                <div class="loading-overlay" id="loadingOverlay">
                    <div class="spinner"></div>
                    <div style="font-size: 0.9rem; font-weight: 600;">Detecting objects & plotting bounding boxes...</div>
                </div>

                <div class="panel-header">
                    <div class="panel-title">
                        <span>📊</span> Detection Results
                    </div>
                    <div class="view-toggle" id="viewToggle" style="display: none;">
                        <button class="view-btn active" id="btnAnnotated" onclick="switchView('annotated')">Boxes</button>
                        <button class="view-btn" id="btnOriginal" onclick="switchView('original')">Original</button>
                    </div>
                </div>

                <!-- Image Preview (With Plotted Bounding Boxes) -->
                <div class="preview-wrapper" id="previewWrapper">
                    <div class="empty-placeholder" id="placeholder">
                        <span>🎯</span>
                        Upload an image or click a sample to detect helmets and traffic violations with bounding boxes
                    </div>
                    <img id="previewImg" class="preview-img" alt="Traffic Violation Detection Preview">
                </div>

                <!-- Alert Banner -->
                <div class="alert-banner" id="alertBanner">
                    <div class="alert-icon" id="alertIcon">⚠️</div>
                    <div class="alert-details">
                        <div class="alert-status" id="alertStatus">TRAFFIC VIOLATION DETECTED</div>
                        <div class="alert-sub" id="alertSub">1 violation identified</div>
                    </div>
                </div>

                <!-- Metrics Grid -->
                <div class="metrics-row" id="metricsRow" style="display: none;">
                    <div class="metric-card">
                        <div class="metric-label">Objects Found</div>
                        <div class="metric-value" id="metricCount">0</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Latency</div>
                        <div class="metric-value" id="metricLatency">-- ms</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Resolution</div>
                        <div class="metric-value" id="metricSize" style="font-size: 0.95rem;">--</div>
                    </div>
                </div>

                <!-- Bounding Boxes List (Detection Mode) -->
                <div id="boxesSection" style="display: none;">
                    <div class="boxes-section-title">
                        <span>📦 Localized Bounding Boxes</span>
                        <span id="boxesCountBadge" style="color: var(--primary);">0 boxes</span>
                    </div>
                    <div class="boxes-list" id="boxesList"></div>
                </div>

                <!-- Probability Distribution (Classification Mode) -->
                <div id="probSection" style="display: none;">
                    <div class="boxes-section-title">Class Probability Distribution</div>
                    <div id="probBars"></div>
                </div>
            </div>

        </div>
    </main>

    <!-- Footer -->
    <footer>
        Traffic Violation Detection System &bull; YOLO11 Bounding Box Object Detection &bull; PyTorch & CUDA
    </footer>

    <!-- Frontend Script -->
    <script>
        let currentMode = 'detect';
        let currentConf = 0.15;
        let currentOriginalSrc = null;
        let currentAnnotatedSrc = null;
        let lastImageFile = null;

        const dropzone = document.getElementById('dropzone');
        const fileInput = document.getElementById('fileInput');
        const previewImg = document.getElementById('previewImg');
        const placeholder = document.getElementById('placeholder');
        const loadingOverlay = document.getElementById('loadingOverlay');
        const alertBanner = document.getElementById('alertBanner');
        const alertIcon = document.getElementById('alertIcon');
        const alertStatus = document.getElementById('alertStatus');
        const alertSub = document.getElementById('alertSub');
        const metricsRow = document.getElementById('metricsRow');
        const metricCount = document.getElementById('metricCount');
        const metricLatency = document.getElementById('metricLatency');
        const metricSize = document.getElementById('metricSize');
        const boxesSection = document.getElementById('boxesSection');
        const boxesList = document.getElementById('boxesList');
        const boxesCountBadge = document.getElementById('boxesCountBadge');
        const probSection = document.getElementById('probSection');
        const probBars = document.getElementById('probBars');
        const viewToggle = document.getElementById('viewToggle');
        const btnAnnotated = document.getElementById('btnAnnotated');
        const btnOriginal = document.getElementById('btnOriginal');
        const confSlider = document.getElementById('confSlider');
        const confVal = document.getElementById('confVal');
        const sliderContainer = document.getElementById('sliderContainer');

        function setMode(mode) {
            currentMode = mode;
            document.getElementById('btnModeDetect').classList.toggle('active', mode === 'detect');
            document.getElementById('btnModeClassify').classList.toggle('active', mode === 'classify');
            sliderContainer.style.display = (mode === 'detect') ? 'flex' : 'none';
            if (lastImageFile) {
                processFile(lastImageFile);
            }
        }

        function updateConf(val) {
            currentConf = val / 100;
            confVal.textContent = `${val}%`;
        }

        confSlider.addEventListener('change', () => {
            if (lastImageFile && currentMode === 'detect') {
                processFile(lastImageFile);
            }
        });

        function switchView(view) {
            if (view === 'annotated' && currentAnnotatedSrc) {
                previewImg.src = currentAnnotatedSrc;
                btnAnnotated.classList.add('active');
                btnOriginal.classList.remove('active');
            } else if (view === 'original' && currentOriginalSrc) {
                previewImg.src = currentOriginalSrc;
                btnOriginal.classList.add('active');
                btnAnnotated.classList.remove('active');
            }
        }

        // Fetch System Info
        fetch('/api/info')
            .then(res => res.json())
            .then(info => {
                document.getElementById('deviceBadge').textContent = `${info.device} Active`;
            })
            .catch(() => {});

        // Load Samples
        fetch('/api/samples')
            .then(res => res.json())
            .then(samples => {
                const grid = document.getElementById('samplesGrid');
                if (!samples || samples.length === 0) {
                    grid.innerHTML = '<span style="font-size:0.75rem; color:#6b7280;">No sample images found.</span>';
                    return;
                }
                grid.innerHTML = '';
                samples.forEach(s => {
                    const card = document.createElement('div');
                    card.className = 'sample-card';
                    card.title = `Test ${s.label}`;
                    card.onclick = () => loadSample(s.path);
                    card.innerHTML = `
                        <img class="sample-thumb" src="${s.path}" alt="${s.label}">
                        <div class="sample-title">${s.label}</div>
                    `;
                    grid.appendChild(card);
                });
            })
            .catch(() => {});

        // Drag & Drop
        ['dragenter', 'dragover'].forEach(eventName => {
            dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                dropzone.classList.add('dragover');
            });
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                dropzone.classList.remove('dragover');
            });
        });

        dropzone.addEventListener('drop', (e) => {
            const files = e.dataTransfer.files;
            if (files && files.length > 0) {
                processFile(files[0]);
            }
        });

        fileInput.addEventListener('change', (e) => {
            if (fileInput.files && fileInput.files.length > 0) {
                processFile(fileInput.files[0]);
            }
        });

        // Clipboard Paste (Ctrl + V)
        window.addEventListener('paste', (e) => {
            const items = (e.clipboardData || e.originalEvent.clipboardData).items;
            for (let item of items) {
                if (item.type.indexOf('image') !== -1) {
                    const blob = item.getAsFile();
                    processFile(blob);
                    break;
                }
            }
        });

        function showLoading(show) {
            loadingOverlay.style.display = show ? 'flex' : 'none';
        }

        function displayPreview(src) {
            placeholder.style.display = 'none';
            previewImg.style.display = 'block';
            previewImg.src = src;
        }

        function processFile(file) {
            if (file.type && !file.type.startsWith('image/')) {
                alert('Please upload a valid image file (PNG, JPG, JPEG).');
                return;
            }

            lastImageFile = file;

            const reader = new FileReader();
            reader.onload = (e) => {
                currentOriginalSrc = e.target.result;
                displayPreview(currentOriginalSrc);
            };
            reader.readAsDataURL(file);

            showLoading(true);
            const formData = new FormData();
            formData.append('file', file, file.name || 'image.jpg');
            formData.append('mode', currentMode);
            formData.append('conf', currentConf);

            fetch('/api/predict', {
                method: 'POST',
                body: formData
            })
            .then(res => {
                if (!res.ok) throw new Error('Prediction failed: ' + res.statusText);
                return res.json();
            })
            .then(data => renderResults(data))
            .catch(err => alert('Detection error: ' + err.message))
            .finally(() => showLoading(false));
        }

        function analyzeUrl() {
            const url = document.getElementById('urlInput').value.trim();
            if (!url) {
                alert('Please enter an image URL.');
                return;
            }

            currentOriginalSrc = url;
            displayPreview(url);
            showLoading(true);

            const formData = new FormData();
            formData.append('url', url);
            formData.append('mode', currentMode);
            formData.append('conf', currentConf);

            fetch('/api/predict-url', {
                method: 'POST',
                body: formData
            })
            .then(res => {
                if (!res.ok) throw new Error('Failed to analyze image from URL');
                return res.json();
            })
            .then(data => renderResults(data))
            .catch(err => alert(err.message))
            .finally(() => showLoading(false));
        }

        function loadSample(sampleUrl) {
            showLoading(true);
            fetch(sampleUrl)
                .then(res => {
                    if (!res.ok) throw new Error('Sample fetch failed');
                    return res.blob();
                })
                .then(blob => {
                    const sampleFile = new File([blob], 'sample.jpg', { type: blob.type || 'image/jpeg' });
                    processFile(sampleFile);
                })
                .catch(err => {
                    showLoading(false);
                    alert('Failed to load sample: ' + err.message);
                });
        }

        function renderResults(data) {
            // Alert Banner
            alertBanner.style.display = 'flex';
            alertBanner.style.background = data.status_bg;
            alertBanner.style.borderColor = data.status_color;
            alertIcon.textContent = data.status_icon;
            alertStatus.textContent = data.status;
            alertStatus.style.color = data.status_color;
            alertSub.textContent = data.status_sub;

            // Metrics
            metricsRow.style.display = 'grid';
            metricLatency.textContent = `${data.inference_ms} ms`;
            metricSize.textContent = data.image_size;

            if (data.mode === 'detect') {
                // Object Detection Mode (Bounding Boxes)
                metricCount.textContent = `${data.total_detections} boxes`;
                probSection.style.display = 'none';
                boxesSection.style.display = 'block';
                viewToggle.style.display = 'flex';

                // Display Annotated Image with plotted boxes!
                if (data.annotated_image) {
                    currentAnnotatedSrc = data.annotated_image;
                    previewImg.src = currentAnnotatedSrc;
                    btnAnnotated.classList.add('active');
                    btnOriginal.classList.remove('active');
                }

                // Render list of bounding boxes
                boxesList.innerHTML = '';
                boxesCountBadge.textContent = `${data.total_detections} boxes`;

                if (data.boxes && data.boxes.length > 0) {
                    data.boxes.forEach(box => {
                        const item = document.createElement('div');
                        item.className = 'box-item';
                        item.style.borderLeftColor = box.color;
                        item.innerHTML = `
                            <div class="box-info">
                                <span>${box.icon}</span>
                                <span style="color: ${box.color};">${box.label}</span>
                                <span class="box-coords">[${box.bbox.join(', ')}]</span>
                            </div>
                            <div class="box-conf" style="color: ${box.color};">
                                ${box.confidence_pct}%
                            </div>
                        `;
                        boxesList.appendChild(item);
                    });
                } else {
                    boxesList.innerHTML = `
                        <div style="font-size: 0.8rem; color: #94a3b8; text-align: center; padding: 1rem;">
                            No objects found above ${Math.round(data.conf_used * 100)}% threshold.<br>
                            <span style="font-size: 0.72rem; color: #64748b;">Try dragging the slider down to lower confidence.</span>
                        </div>
                    `;
                }
            } else {
                // Classification Mode
                metricCount.textContent = '1 Class';
                boxesSection.style.display = 'none';
                probSection.style.display = 'block';
                viewToggle.style.display = 'none';

                probBars.innerHTML = '';
                const CLASS_COLORS = {
                    'no_helmet': '#ef4444',
                    'overloading': '#dc2626',
                    'helmet': '#10b981',
                    'blur': '#f59e0b'
                };

                for (const [cls, pct] of Object.entries(data.all_probabilities)) {
                    const color = CLASS_COLORS[cls] || '#3b82f6';
                    const formattedName = cls.replace('_', ' ').toUpperCase();

                    const item = document.createElement('div');
                    item.className = 'prob-item';
                    item.innerHTML = `
                        <div class="prob-header">
                            <span style="color: ${cls === data.class ? '#ffffff' : '#94a3b8'}; font-weight: ${cls === data.class ? '700' : '500'};">
                                ${cls === data.class ? '👉 ' : ''}${formattedName}
                            </span>
                            <span style="font-family: 'JetBrains Mono', monospace; color: ${color};">
                                ${pct.toFixed(1)}%
                            </span>
                        </div>
                        <div class="prob-track">
                            <div class="prob-fill" style="background: ${color}; width: ${pct}%;"></div>
                        </div>
                    `;
                    probBars.appendChild(item);
                }
            }
        }
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    port = 8000
    print("=" * 60)
    print("  TRAFFIC VIOLATION DETECTION - OBJECT DETECTION WEB APP")
    print(f"  Running locally at: http://127.0.0.1:{port}")
    if DETECTION_MODEL_PATH:
        print(f"  Detection Model     : {DETECTION_MODEL_PATH.name}")
    if CLASSIFICATION_MODEL_PATH:
        print(f"  Classification Model: {CLASSIFICATION_MODEL_PATH.name}")
    print(f"  Hardware Device     : {DEVICE_NAME}")
    print("=" * 60)
    uvicorn.run("app:app", host="127.0.0.1", port=port, reload=False)
