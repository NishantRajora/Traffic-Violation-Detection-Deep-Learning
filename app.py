"""
========================================================================================
AI Traffic Guard - White Cathedral Edition
Real-Time Helmet Detection, Video Stream & Traffic Violation Analytics Dashboard
========================================================================================
Advanced Computer Vision Web Application powered by FastAPI & YOLOv8:
  - Real-time helmet compliance and violation monitoring
  - AuthKit Style: Clean White Cathedral Edition
      * Clean white canvas (#ffffff / #f8fafc) with subtle 80px blueprint grid & ambient violet halo
      * Crisp white frosted glass surfaces with subtle borders & soft elevation
      * Slate & Violet gradient wordmarks and tracked DotDigital all-caps eyebrow labels
      * Void Violet (#663af3) primary CTAs, Ember Red (#dc2626) violations, Deep Teal (#059669) compliance
      * One-click Light / Dark Mode theme toggle
  - Features:
      * Tab 1: Image Inspector (Drag & Drop, File Upload, Clipboard Paste Ctrl+V, Quick Test Gallery)
      * Tab 2: YouTube Video Inspector (Extract & Analyze frames directly from YouTube URLs)
      * Tab 3: Video File Upload Analyzer (Analyze MP4/AVI/MOV traffic video clips by timestamp)
      * Tab 4: Live WebCam Stream HUD (Real-time video feed streaming to YOLOv8 with FPS counter)
      * Tab 5: Direct Image URL Stream Analyzer
      * Tab 6: Incident Analytics & Session Log with CSV/JSON Export & Historical Telemetry
      * Side-by-Side (Original vs Annotated) and full telemetry breakdown table
      * Audio Alert Chime for detected violations
========================================================================================
"""

import os
# Fix OpenMP duplicate library conflict on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import base64
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import torch
from ultralytics import YOLO
from fastapi import FastAPI, UploadFile, File, Form, Body, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
import uvicorn
import yt_dlp

# --- CONFIGURATION & PATHS ---
PROJECT_ROOT = Path(__file__).resolve().parent

# Auto-locate model weights
CANDIDATE_WEIGHTS = [
    PROJECT_ROOT / "saved_models" / "best.pt",
    PROJECT_ROOT / "runs" / "detect" / "train" / "weights" / "best.pt",
    PROJECT_ROOT / "runs" / "detect" / "runs" / "detect" / "train" / "weights" / "best.pt",
    PROJECT_ROOT / "yolov8n.pt"
]

MODEL_PATH = None
for p in CANDIDATE_WEIGHTS:
    if p.exists():
        MODEL_PATH = p
        break

# Device Selection
DEVICE = 0 if torch.cuda.is_available() else "cpu"
DEVICE_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

# Sample Images Directories
SAMPLE_DIRS = [
    PROJECT_ROOT / "HelmetDataset" / "test" / "images",
    PROJECT_ROOT / "data" / "images"
]

# Load YOLO Model
model = None
if MODEL_PATH:
    try:
        model = YOLO(str(MODEL_PATH))
        print(f"🚀 Loaded Model Weights: {MODEL_PATH} on {DEVICE_NAME}")
    except Exception as e:
        print(f"❌ Failed to load model from {MODEL_PATH}: {e}")

# Color mappings (BGR)
# Deep Emerald/Teal #059669 -> BGR: (105, 150, 5)
COLOR_SAFE = (105, 150, 5)
# Vivid Red #dc2626 -> BGR: (38, 38, 220)
COLOR_VIOLATION = (38, 38, 220)
COLOR_DEFAULT = (250, 228, 209)

# In-memory detection history log
DETECTION_HISTORY = []

app = FastAPI(
    title="AI Traffic Guard - White Cathedral Edition",
    version="3.1"
)


# ======================================================================================
# INFERENCE HELPER FUNCTIONS
# ======================================================================================

def annotate_frame(image_bgr: np.ndarray, results, conf_threshold: float = 0.25):
    """
    Draws custom high-contrast bounding boxes, label badges, and violation HUD.
    """
    annotated = image_bgr.copy()
    h, w = annotated.shape[:2]

    stats = {"With Helmet": 0, "Without Helmet": 0}
    detections = []

    for r in results:
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            continue

        for i, box in enumerate(boxes):
            conf = float(box.conf[0])
            if conf < conf_threshold:
                continue

            cls_id = int(box.cls[0])
            cls_name = r.names.get(cls_id, str(cls_id))

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            is_violation = (cls_name.lower() == "without helmet")
            color = COLOR_VIOLATION if is_violation else COLOR_SAFE
            stats[cls_name] = stats.get(cls_name, 0) + 1

            detections.append({
                "id": i + 1,
                "class": cls_name,
                "confidence": round(conf * 100, 1),
                "is_violation": is_violation,
                "box": [x1, y1, x2, y2],
                "width": x2 - x1,
                "height": y2 - y1
            })

            # Bounding Box with sleek hairline & corner brackets
            thickness = max(2, int(min(w, h) / 360))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            # High-tech corner brackets
            c_len = max(10, int(min(x2 - x1, y2 - y1) * 0.22))
            c_thick = thickness + 2
            cv2.line(annotated, (x1, y1), (x1 + c_len, y1), color, c_thick)
            cv2.line(annotated, (x1, y1), (x1, y1 + c_len), color, c_thick)
            cv2.line(annotated, (x2, y1), (x2 - c_len, y1), color, c_thick)
            cv2.line(annotated, (x2, y1), (x2, y1 + c_len), color, c_thick)
            cv2.line(annotated, (x1, y2), (x1 + c_len, y2), color, c_thick)
            cv2.line(annotated, (x1, y2), (x1, y2 - c_len), color, c_thick)
            cv2.line(annotated, (x2, y2), (x2 - c_len, y2), color, c_thick)
            cv2.line(annotated, (x2, y2), (x2, y2 - c_len), color, c_thick)

            # Frosted Tag with high-visibility styling
            label_text = f" {cls_name.upper()}  {conf * 100:.1f}% "
            font_scale = max(0.45, min(w, h) / 1050)
            (lw, lh), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            y_tag = max(0, y1 - lh - 10)

            # Badge background with frosted accent
            cv2.rectangle(annotated, (x1, y_tag), (x1 + lw + 6, y_tag + lh + 10), (15, 10, 5), -1)
            cv2.rectangle(annotated, (x1, y_tag), (x1 + lw + 6, y_tag + lh + 10), color, 1)
            cv2.putText(
                annotated,
                label_text,
                (x1 + 3, y_tag + lh + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

    # Top Violation Status Banner
    has_violations = stats.get("Without Helmet", 0) > 0
    safe_count = stats.get("With Helmet", 0)
    viol_count = stats.get("Without Helmet", 0)
    total_count = safe_count + viol_count

    banner_bg = (15, 10, 5)
    banner_accent = COLOR_VIOLATION if has_violations else COLOR_SAFE
    banner_text = f" VIOLATIONS: {viol_count}  |  SAFE (HELMET): {safe_count}  |  TOTAL: {total_count} "

    (bw, bh), _ = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, bh + 24), banner_bg, -1)
    cv2.addWeighted(overlay, 0.85, annotated, 0.15, 0, annotated)

    cv2.rectangle(annotated, (0, 0), (6, bh + 24), banner_accent, -1)
    cv2.putText(
        annotated,
        banner_text,
        (16, bh + 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (240, 245, 255),
        2,
        cv2.LINE_AA
    )

    return annotated, stats, detections


def run_pipeline(img_bgr: np.ndarray, conf: float = 0.25, iou: float = 0.45, source_label: str = "Image"):
    """Executes YOLOv8 detection pipeline and packages response."""
    if model is None:
        raise RuntimeError("YOLO model is not initialized on the server.")

    t0 = time.time()
    results = model.predict(img_bgr, conf=conf, iou=iou, device=DEVICE, verbose=False)[0]
    latency_ms = round((time.time() - t0) * 1000, 1)

    annotated, stats, detections = annotate_frame(img_bgr, [results], conf_threshold=conf)

    safe_count = stats.get("With Helmet", 0)
    viol_count = stats.get("Without Helmet", 0)
    total_riders = safe_count + viol_count
    compliance = round((safe_count / total_riders * 100), 1) if total_riders > 0 else 100.0

    # Encode annotated and original to base64
    _, buf_ann = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
    _, buf_orig = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])

    b64_annotated = base64.b64encode(buf_ann).decode('utf-8')
    b64_original = base64.b64encode(buf_orig).decode('utf-8')

    response_data = {
        "success": True,
        "source": source_label,
        "with_helmet": safe_count,
        "without_helmet": viol_count,
        "total_riders": total_riders,
        "compliance_rate": compliance,
        "violation": viol_count > 0,
        "image_url": f"data:image/jpeg;base64,{b64_annotated}",
        "original_url": f"data:image/jpeg;base64,{b64_original}",
        "detections": detections,
        "latency_ms": latency_ms,
        "device": DEVICE_NAME,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # Record in history (limit to 40 items)
    DETECTION_HISTORY.insert(0, {
        "id": len(DETECTION_HISTORY) + 1,
        "source": source_label,
        "timestamp": response_data["timestamp"],
        "safe": safe_count,
        "violations": viol_count,
        "total": total_riders,
        "compliance": compliance,
        "status": "VIOLATION" if viol_count > 0 else "COMPLIANT",
        "latency_ms": latency_ms,
        "thumbnail": f"data:image/jpeg;base64,{b64_annotated}"
    })
    if len(DETECTION_HISTORY) > 40:
        DETECTION_HISTORY.pop()

    return response_data


def extract_youtube_frame(youtube_url: str, timestamp_sec: float = 0.0):
    """
    Extracts a frame and metadata from a YouTube video URL using yt-dlp and OpenCV.
    Uses mobile and embedded player clients to bypass YouTube sign-in bot restrictions.
    """
    ydl_opts = {
        'format': 'best[ext=mp4]/best',
        'quiet': True,
        'no_warnings': True,
        'extractor_args': {
            'youtube': {
                'player_client': ['android', 'ios', 'web_embedded']
            }
        }
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        stream_url = info.get('url')
        title = info.get('title', 'YouTube Traffic Stream')
        duration = info.get('duration', 0)
        thumbnail = info.get('thumbnail', '')

        if not stream_url:
            raise ValueError("Direct video stream URL could not be resolved from YouTube.")

        cap = cv2.VideoCapture(stream_url)
        if timestamp_sec > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(timestamp_sec)) * 1000.0)

        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            # Fallback to high-res thumbnail if stream frame read fails
            if thumbnail:
                req = urllib.request.Request(thumbnail, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    arr = np.asarray(bytearray(resp.read()), dtype=np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

            if frame is None:
                raise ValueError("Could not read frame from video stream.")

        return frame, title, duration


# ======================================================================================
# FRONTEND (HTML + CSS + JS) - CLEAN WHITE CATHEDRAL EDITION
# ======================================================================================

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Traffic Guard — Helmet Detection & Analytics</title>
    <!-- Bootstrap 5 CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <!-- FontAwesome 6 Icons -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
    <!-- Google Fonts: Space Grotesk, Inter, JetBrains Mono -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">

    <style>
        /* ==========================================================================
           COLOR TOKENS & WHITE THEME CONFIGURATION
           ========================================================================== */
        :root {
            /* Light White Mode (Default) */
            --color-canvas: #ffffff;
            --color-canvas-subtle: #f8fafc;
            --color-surface: #ffffff;
            --color-surface-subtle: #f8fafc;
            --color-border: #e2e8f0;
            --color-border-hover: #cbd5e1;
            --color-text-primary: #0f172a;
            --color-text-secondary: #475569;
            --color-text-muted: #64748b;
            --color-eyebrow: #64748b;
            --color-nav-bg: rgba(255, 255, 255, 0.94);
            
            /* Accents */
            --color-void-violet: #663af3;
            --color-void-violet-hover: #5429db;
            --color-blueprint-blue: #3b82f6;
            --color-ember-glow: #dc2626;
            --color-deep-teal: #059669;
            --color-amber-compliance: #4f46e5;
            --color-luminous-fill: rgba(102, 58, 243, 0.05);

            /* Typography */
            --font-aeonikpro: 'Space Grotesk', -apple-system, sans-serif;
            --font-untitled-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            --font-dotdigital: 'JetBrains Mono', monospace;

            /* Radii */
            --radius-pill: 999px;
            --radius-card: 16px;
            --radius-badge: 6px;
            --radius-circle: 9999px;

            /* Shadows */
            --shadow-card: 0 4px 20px rgba(0, 0, 0, 0.04), 0 1px 3px rgba(0, 0, 0, 0.02);
            --shadow-button-primary: 0 4px 16px rgba(102, 58, 243, 0.28);
        }

        /* Dark Theme Override */
        [data-theme="dark"] {
            --color-canvas: #05060f;
            --color-canvas-subtle: #090c17;
            --color-surface: rgba(186, 214, 247, 0.035);
            --color-surface-subtle: rgba(47, 52, 62, 0.45);
            --color-border: rgba(186, 215, 247, 0.12);
            --color-border-hover: rgba(186, 215, 247, 0.28);
            --color-text-primary: #ffffff;
            --color-text-secondary: #c7d3ea;
            --color-text-muted: #9da7ba;
            --color-eyebrow: #c7d3ea;
            --color-nav-bg: rgba(5, 6, 15, 0.88);
            --color-luminous-fill: rgba(199, 211, 234, 0.08);
            --shadow-card: 0 16px 36px rgba(5, 6, 15, 0.85), inset 0 1px 1px rgba(216, 236, 248, 0.15);
        }

        /* Base Body with Clean White Canvas */
        body {
            background-color: var(--color-canvas);
            background-image: 
                radial-gradient(ellipse 90% 45% at 50% -10%, rgba(102, 58, 243, 0.04), transparent 70%),
                linear-gradient(to right, rgba(0, 0, 0, 0.035) 1px, transparent 1px),
                linear-gradient(to bottom, rgba(0, 0, 0, 0.035) 1px, transparent 1px);
            background-size: 100% 100%, 80px 80px, 80px 80px;
            color: var(--color-text-primary);
            font-family: var(--font-untitled-sans);
            min-height: 100vh;
            overflow-x: hidden;
            -webkit-font-smoothing: antialiased;
            transition: background-color 0.3s ease, color 0.3s ease;
        }

        [data-theme="dark"] body {
            background-image: 
                radial-gradient(ellipse 90% 50% at 50% -12%, rgba(124, 145, 182, 0.22), transparent 70%),
                linear-gradient(to right, rgba(186, 215, 247, 0.045) 1px, transparent 1px),
                linear-gradient(to bottom, rgba(186, 215, 247, 0.045) 1px, transparent 1px);
        }

        /* Navigation Bar */
        .cathedral-nav {
            background: var(--color-nav-bg);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border-bottom: 1px solid var(--color-border);
            position: sticky;
            top: 0;
            z-index: 1000;
            transition: background 0.3s ease, border-color 0.3s ease;
        }

        .brand-wordmark {
            font-family: var(--font-aeonikpro);
            font-weight: 700;
            font-size: 1.35rem;
            letter-spacing: -0.02em;
            background: linear-gradient(135deg, var(--color-text-primary) 0%, var(--color-void-violet) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        /* DotDigital Eyebrow Label */
        .eyebrow-label {
            font-family: var(--font-dotdigital);
            font-size: 0.72rem;
            letter-spacing: 0.10em;
            text-transform: uppercase;
            color: var(--color-eyebrow);
        }

        /* Cards & Glass Plates */
        .glass-plate {
            background: var(--color-surface);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid var(--color-border);
            border-radius: var(--radius-card);
            box-shadow: var(--shadow-card);
            transition: border-color 0.25s ease, box-shadow 0.25s ease, background-color 0.3s ease;
        }
        .glass-plate:hover {
            border-color: var(--color-border-hover);
        }

        /* Subtle Inner Subplates */
        .steel-subplate {
            background: var(--color-surface-subtle);
            border: 1px solid var(--color-border);
            border-radius: 12px;
            transition: background-color 0.3s ease, border-color 0.3s ease;
        }

        /* Buttons */
        .btn-void-violet {
            background: var(--color-void-violet);
            color: #ffffff !important;
            border-radius: var(--radius-pill);
            padding: 9px 24px;
            font-weight: 500;
            font-size: 0.88rem;
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: var(--shadow-button-primary);
            transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
        }
        .btn-void-violet:hover {
            background: var(--color-void-violet-hover);
            color: #ffffff !important;
            transform: translateY(-1px);
            box-shadow: 0 6px 20px rgba(102, 58, 243, 0.45);
        }
        .btn-void-violet:disabled {
            opacity: 0.5;
            transform: none;
            cursor: not-allowed;
        }

        .btn-ghost-pill {
            background: var(--color-surface-subtle);
            color: var(--color-text-secondary);
            border: 1px solid var(--color-border);
            border-radius: var(--radius-pill);
            padding: 8px 18px;
            font-size: 0.85rem;
            font-weight: 500;
            transition: all 0.2s ease;
        }
        .btn-ghost-pill:hover, .btn-ghost-pill.active {
            background: var(--color-border);
            color: var(--color-text-primary);
            border-color: var(--color-border-hover);
        }

        /* Badges */
        .auth-badge {
            background: var(--color-surface-subtle);
            border: 1px solid var(--color-border);
            border-radius: var(--radius-badge);
            padding: 4px 10px;
            font-family: var(--font-dotdigital);
            font-size: 0.72rem;
            color: var(--color-text-muted);
        }
        .auth-badge.live-teal {
            color: var(--color-deep-teal);
            border-color: rgba(5, 150, 105, 0.35);
            background: rgba(5, 150, 105, 0.1);
        }

        /* Navigation Tab Pills */
        .cathedral-tabs .nav-link {
            color: var(--color-text-muted);
            border-radius: var(--radius-pill);
            padding: 9px 20px;
            font-size: 0.85rem;
            font-weight: 500;
            border: 1px solid transparent;
            transition: all 0.2s ease;
        }
        .cathedral-tabs .nav-link:hover {
            color: var(--color-text-primary);
            background: var(--color-surface-subtle);
        }
        .cathedral-tabs .nav-link.active {
            color: #ffffff !important;
            background: var(--color-void-violet);
            border: 1px solid var(--color-void-violet);
            box-shadow: 0 4px 14px rgba(102, 58, 243, 0.25);
        }

        /* Drop Zone */
        .cathedral-dropzone {
            border: 1.5px dashed var(--color-border-hover);
            background: var(--color-surface-subtle);
            border-radius: 14px;
            padding: 38px 20px;
            text-align: center;
            cursor: pointer;
            transition: all 0.25s ease;
        }
        .cathedral-dropzone:hover, .cathedral-dropzone.dragover {
            border-color: var(--color-void-violet);
            background: rgba(102, 58, 243, 0.05);
            box-shadow: 0 0 20px rgba(102, 58, 243, 0.15);
            transform: translateY(-2px);
        }

        /* Telemetry Stat Cards */
        .telemetry-card {
            background: var(--color-surface);
            border: 1px solid var(--color-border);
            border-radius: 12px;
            padding: 14px 16px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03);
            position: relative;
            transition: background-color 0.3s ease, border-color 0.3s ease;
        }
        .telemetry-card.safe { border-left: 3px solid var(--color-deep-teal); }
        .telemetry-card.violation { border-left: 3px solid var(--color-ember-glow); }
        .telemetry-card.compliance { border-left: 3px solid var(--color-amber-compliance); }
        .telemetry-card.latency { border-left: 3px solid var(--color-text-muted); }

        .telemetry-number {
            font-family: var(--font-dotdigital);
            font-size: 1.85rem;
            font-weight: 600;
            line-height: 1.1;
        }

        /* Status Banner Pill */
        .cathedral-status-pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 9px 24px;
            border-radius: var(--radius-pill);
            font-family: var(--font-dotdigital);
            font-size: 0.85rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            transition: all 0.3s ease;
        }
        .cathedral-status-pill.safe {
            background: rgba(5, 150, 105, 0.1);
            border: 1px solid var(--color-deep-teal);
            color: var(--color-deep-teal);
            box-shadow: 0 0 20px rgba(5, 150, 105, 0.15);
        }
        .cathedral-status-pill.violation {
            background: rgba(220, 38, 38, 0.12);
            border: 1px solid var(--color-ember-glow);
            color: var(--color-ember-glow);
            box-shadow: 0 0 24px rgba(220, 38, 38, 0.2);
            animation: cathedral-pulse 2s infinite;
        }
        @keyframes cathedral-pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.02); }
            100% { transform: scale(1); }
        }

        /* Viewport Frame for Video & Images */
        .viewport-frame {
            border-radius: 14px;
            overflow: hidden;
            background: #0b0f19;
            border: 1px solid var(--color-border);
            position: relative;
            box-shadow: 0 12px 36px rgba(0, 0, 0, 0.18);
            min-height: 380px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .viewport-frame img {
            width: 100%;
            height: auto;
            max-height: 560px;
            object-fit: contain;
            display: block;
        }

        /* Form Inputs */
        .auth-input {
            background: var(--color-surface);
            border: 1px solid var(--color-border-hover);
            border-radius: var(--radius-badge);
            color: var(--color-text-primary);
            font-family: var(--font-untitled-sans);
            padding: 9px 14px;
            font-size: 0.88rem;
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
        }
        .auth-input:focus {
            background: var(--color-surface);
            border-color: var(--color-void-violet);
            box-shadow: 0 0 0 3px rgba(102, 58, 243, 0.15);
            color: var(--color-text-primary);
            outline: none;
        }

        /* Range Slider */
        .form-range::-webkit-slider-thumb {
            background: var(--color-void-violet);
            box-shadow: 0 0 10px var(--color-void-violet);
            border: 1px solid #ffffff;
        }

        /* Sample Gallery Item */
        .sample-tile {
            border-radius: 10px;
            overflow: hidden;
            cursor: pointer;
            border: 1px solid var(--color-border);
            transition: all 0.2s ease;
            background: var(--color-surface);
        }
        .sample-tile:hover {
            transform: translateY(-2px);
            border-color: var(--color-void-violet);
            box-shadow: 0 6px 16px rgba(102, 58, 243, 0.15);
        }
        .sample-tile img {
            width: 100%;
            height: 85px;
            object-fit: cover;
        }

        /* Table Styling */
        .cathedral-table {
            --bs-table-bg: transparent;
            --bs-table-color: var(--color-text-primary);
            font-size: 0.85rem;
        }
        .cathedral-table th {
            color: var(--color-text-muted);
            font-family: var(--font-dotdigital);
            font-weight: 500;
            letter-spacing: 0.05em;
            font-size: 0.75rem;
            border-bottom: 1px solid var(--color-border);
            background: var(--color-surface-subtle);
            padding: 10px 12px;
        }
        .cathedral-table td {
            border-bottom: 1px solid var(--color-border);
            padding: 10px 12px;
            vertical-align: middle;
            color: var(--color-text-secondary);
        }

        /* Spinner */
        .cathedral-spinner {
            width: 44px;
            height: 44px;
            border: 3px solid rgba(102, 58, 243, 0.2);
            border-top: 3px solid var(--color-void-violet);
            border-radius: 50%;
            animation: cathedral-spin 0.8s linear infinite;
            margin: 30px auto;
        }
        @keyframes cathedral-spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }

        /* Scrollbar */
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: var(--color-canvas); }
        ::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: var(--color-void-violet); }
    </style>
</head>
<body>

    <!-- ==========================================================================
         TOP NAVIGATION BAR
         ========================================================================== -->
    <nav class="navbar navbar-expand-lg cathedral-nav px-4 py-2">
        <div class="container-fluid">
            <a class="navbar-brand d-flex align-items-center gap-2" href="#">
                <i class="fas fa-shield-halved fs-3" style="color: var(--color-void-violet);"></i>
                <div>
                    <span class="brand-wordmark">AUTHKIT GUARD</span>
                    <span class="d-block eyebrow-label" style="font-size: 0.62rem;">REAL-TIME VIOLATION ENGINE &bull; YOLOV8</span>
                </div>
            </a>

            <!-- System Spec Badges & Controls -->
            <div class="d-flex align-items-center gap-2">
                <span class="auth-badge live-teal" id="nav-device-badge">
                    <i class="fas fa-microchip me-1"></i> <span id="nav-device">ACCELERATION ACTIVE</span>
                </span>
                <span class="auth-badge">
                    <i class="fas fa-cube me-1"></i> YOLOv8n (2 Classes)
                </span>
                <button class="btn-ghost-pill btn-sm" onclick="toggleAudioAlert()" id="btn-audio">
                    <i class="fas fa-volume-high me-1"></i> Sound: ON
                </button>
                <!-- Theme Toggle Button -->
                <button class="btn-ghost-pill btn-sm" onclick="toggleTheme()" id="btn-theme" title="Toggle Light / Dark Mode">
                    <i class="fas fa-moon me-1"></i> Dark Mode
                </button>
            </div>
        </div>
    </nav>

    <!-- ==========================================================================
         MAIN DASHBOARD
         ========================================================================== -->
    <div class="container py-4">

        <!-- Tab Selection Header -->
        <div class="d-flex justify-content-center mb-4">
            <ul class="nav nav-pills cathedral-tabs gap-2" id="dashboardTabs" role="tablist">
                <li class="nav-item">
                    <button class="nav-link active" id="tab-image-btn" data-bs-toggle="pill" data-bs-target="#tab-image" type="button">
                        <i class="fas fa-image me-1"></i> Image Inspector
                    </button>
                </li>
                <li class="nav-item">
                    <button class="nav-link" id="tab-yt-btn" data-bs-toggle="pill" data-bs-target="#tab-yt" type="button">
                        <i class="fab fa-youtube me-1" style="color: #ff0033;"></i> YouTube Stream
                    </button>
                </li>
                <li class="nav-item">
                    <button class="nav-link" id="tab-video-btn" data-bs-toggle="pill" data-bs-target="#tab-video" type="button">
                        <i class="fas fa-film me-1"></i> Video Upload
                    </button>
                </li>
                <li class="nav-item">
                    <button class="nav-link" id="tab-webcam-btn" data-bs-toggle="pill" data-bs-target="#tab-webcam" type="button">
                        <i class="fas fa-video me-1"></i> Live WebCam
                    </button>
                </li>
                <li class="nav-item">
                    <button class="nav-link" id="tab-url-btn" data-bs-toggle="pill" data-bs-target="#tab-url" type="button">
                        <i class="fas fa-globe me-1"></i> Image URL
                    </button>
                </li>
                <li class="nav-item">
                    <button class="nav-link" id="tab-history-btn" data-bs-toggle="pill" data-bs-target="#tab-history" type="button" onclick="loadHistoryTable()">
                        <i class="fas fa-chart-line me-1"></i> Incident Log
                    </button>
                </li>
            </ul>
        </div>

        <!-- ======================================================================
             TAB CONTENT PANELS
             ====================================================================== -->
        <div class="tab-content" id="dashboardTabContent">

            <!-- ------------------------------------------------------------------
                 TAB 1: IMAGE INSPECTOR
                 ------------------------------------------------------------------ -->
            <div class="tab-pane fade show active" id="tab-image" role="tabpanel">
                <div class="row g-4">
                    <!-- Left Config Controls -->
                    <div class="col-lg-5">
                        <div class="glass-plate p-4 h-100">
                            <div class="d-flex justify-content-between align-items-center mb-3">
                                <span class="eyebrow-label"><i class="fas fa-sliders me-1"></i> Detection Parameters</span>
                                <span class="auth-badge">YOLOv8</span>
                            </div>

                            <!-- Drop Zone -->
                            <div id="drop-zone" class="cathedral-dropzone mb-3" onclick="document.getElementById('fileInput').click()">
                                <i class="fas fa-cloud-arrow-up fa-2x mb-2" style="color: var(--color-void-violet);"></i>
                                <h6 class="fw-semibold mb-1" style="font-family: var(--font-aeonikpro); color: var(--color-text-primary);">Drag & Drop Image or Click</h6>
                                <p class="small mb-0" style="color: var(--color-text-muted);">Supports JPG, PNG, WEBP &bull; <b>Ctrl+V</b> to Paste</p>
                                <input type="file" id="fileInput" hidden accept="image/*" onchange="handleFileUpload(this.files[0])">
                            </div>

                            <!-- Sliders -->
                            <div class="steel-subplate p-3 mb-3">
                                <div class="d-flex justify-content-between small mb-1">
                                    <span style="color: var(--color-text-secondary);">Confidence Threshold</span>
                                    <span id="conf-val" class="font-monospace fw-bold" style="color: var(--color-void-violet);">25%</span>
                                </div>
                                <input type="range" class="form-range" id="conf-slider" min="10" max="90" value="25" oninput="updateConfidence(this.value)">
                            </div>

                            <div class="steel-subplate p-3 mb-4">
                                <div class="d-flex justify-content-between small mb-1">
                                    <span style="color: var(--color-text-secondary);">IoU / NMS Suppression</span>
                                    <span id="iou-val" class="font-monospace fw-bold" style="color: var(--color-void-violet);">45%</span>
                                </div>
                                <input type="range" class="form-range" id="iou-slider" min="10" max="80" value="45" oninput="document.getElementById('iou-val').innerText = this.value + '%'">
                            </div>

                            <!-- Quick Sample Gallery -->
                            <div>
                                <div class="d-flex justify-content-between align-items-center mb-2">
                                    <span class="eyebrow-label"><i class="fas fa-images me-1"></i> Test Road Scenes</span>
                                    <button class="btn btn-link btn-sm p-0 text-decoration-none" style="color: var(--color-void-violet);" onclick="loadSampleGallery()">
                                        <i class="fas fa-arrows-rotate"></i> Refresh
                                    </button>
                                </div>
                                <div class="row g-2" id="sample-gallery-container">
                                    <div class="small text-center py-2" style="color: var(--color-text-muted);">Loading sample scenes...</div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Right Telemetry & Viewport -->
                    <div class="col-lg-7">
                        <div class="glass-plate p-4">
                            <!-- Metrics Cards Grid -->
                            <div class="row g-3 mb-3">
                                <div class="col-6 col-md-3">
                                    <div class="telemetry-card safe">
                                        <div class="eyebrow-label" style="font-size: 0.65rem;">Safe Riders</div>
                                        <div class="telemetry-number" style="color: var(--color-deep-teal);" id="stat-safe">0</div>
                                    </div>
                                </div>
                                <div class="col-6 col-md-3">
                                    <div class="telemetry-card violation">
                                        <div class="eyebrow-label" style="font-size: 0.65rem;">Violations</div>
                                        <div class="telemetry-number" style="color: var(--color-ember-glow);" id="stat-violation">0</div>
                                    </div>
                                </div>
                                <div class="col-6 col-md-3">
                                    <div class="telemetry-card compliance">
                                        <div class="eyebrow-label" style="font-size: 0.65rem;">Compliance</div>
                                        <div class="telemetry-number" style="color: var(--color-amber-compliance);" id="stat-compliance">100%</div>
                                    </div>
                                </div>
                                <div class="col-6 col-md-3">
                                    <div class="telemetry-card latency">
                                        <div class="eyebrow-label" style="font-size: 0.65rem;">Latency</div>
                                        <div class="telemetry-number" style="color: var(--color-text-primary);" id="stat-latency">--<small style="font-size:0.75rem">ms</small></div>
                                    </div>
                                </div>
                            </div>

                            <!-- Status Pill -->
                            <div class="text-center my-3">
                                <div id="status-pill" class="cathedral-status-pill safe">
                                    <i class="fas fa-shield-check"></i> SYSTEM READY &bull; WAITING FOR INPUT
                                </div>
                            </div>

                            <!-- View Controls -->
                            <div class="d-flex justify-content-between align-items-center mb-2 px-1">
                                <div class="btn-group btn-group-sm" role="group">
                                    <button type="button" class="btn-ghost-pill active" id="btn-view-annotated" onclick="switchView('annotated')">Annotated</button>
                                    <button type="button" class="btn-ghost-pill" id="btn-view-side" onclick="switchView('side')">Side-by-Side</button>
                                    <button type="button" class="btn-ghost-pill" id="btn-view-original" onclick="switchView('original')">Original</button>
                                </div>
                                <div class="d-flex gap-2">
                                    <button class="btn-ghost-pill" onclick="downloadAnnotatedImage()" title="Download HD Result">
                                        <i class="fas fa-download"></i> Save
                                    </button>
                                    <button class="btn-ghost-pill" onclick="copyDetectionsJSON()" title="Copy JSON Telemetry">
                                        <i class="fas fa-code"></i> JSON
                                    </button>
                                </div>
                            </div>

                            <!-- Image Viewport -->
                            <div class="viewport-frame" id="viewport-container">
                                <div id="loader" class="cathedral-spinner" style="display: none;"></div>
                                <img id="viewport-img" src="" alt="Traffic Scene" style="display: none;">
                                
                                <!-- Side by Side Container -->
                                <div id="side-by-side-container" class="row g-2 w-100 p-2" style="display: none;">
                                    <div class="col-6 text-center">
                                        <div class="eyebrow-label mb-1 text-light">ORIGINAL SCENE</div>
                                        <img id="side-img-orig" src="" style="width: 100%; border-radius: 8px;">
                                    </div>
                                    <div class="col-6 text-center">
                                        <div class="eyebrow-label mb-1 text-light">YOLOV8 ANNOTATED</div>
                                        <img id="side-img-ann" src="" style="width: 100%; border-radius: 8px;">
                                    </div>
                                </div>

                                <div id="viewport-placeholder" class="text-center py-5 text-light opacity-75">
                                    <i class="fas fa-road fa-3x mb-3 text-secondary"></i>
                                    <h6 style="font-family: var(--font-aeonikpro); color: #ffffff;">No Inspection Active</h6>
                                    <p class="small mb-0 text-secondary">Upload an image, inspect a YouTube stream, or pick a test scene from the left.</p>
                                </div>
                            </div>

                            <!-- Detected Bounding Boxes Breakdown Table -->
                            <div class="mt-4" id="table-container" style="display: none;">
                                <div class="d-flex justify-content-between align-items-center mb-2">
                                    <span class="eyebrow-label"><i class="fas fa-list-check me-1"></i> Coordinate & Confidence Telemetry</span>
                                    <span class="auth-badge" id="objects-count-badge">0 objects</span>
                                </div>
                                <div class="table-responsive" style="max-height: 220px;">
                                    <table class="table cathedral-table">
                                        <thead>
                                            <tr>
                                                <th>#</th>
                                                <th>Classification</th>
                                                <th>Confidence</th>
                                                <th>Bounding Box [x1, y1, x2, y2]</th>
                                                <th>Compliance</th>
                                            </tr>
                                        </thead>
                                        <tbody id="detections-tbody"></tbody>
                                    </table>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- ------------------------------------------------------------------
                 TAB 2: YOUTUBE VIDEO STREAM ANALYZER
                 ------------------------------------------------------------------ -->
            <div class="tab-pane fade" id="tab-yt" role="tabpanel">
                <div class="glass-plate p-4">
                    <div class="d-flex justify-content-between align-items-center mb-3">
                        <div>
                            <span class="eyebrow-label"><i class="fab fa-youtube text-danger me-1"></i> YouTube Video Stream Frame Inspector</span>
                            <h5 class="fw-bold mb-0" style="font-family: var(--font-aeonikpro); color: var(--color-text-primary);">Analyze Online YouTube Traffic Feeds</h5>
                        </div>
                        <span class="auth-badge"><i class="fas fa-bolt me-1"></i> yt-dlp + YOLOv8</span>
                    </div>

                    <p class="small mb-4" style="color: var(--color-text-muted);">
                        Extracts high-resolution video frames directly from any YouTube traffic or road surveillance clip and executes instant YOLOv8 helmet detection.
                    </p>

                    <div class="row g-3 mb-4">
                        <div class="col-lg-8">
                            <label class="eyebrow-label mb-1">YouTube Video URL</label>
                            <input type="text" id="yt-url-input" class="form-control auth-input w-100" 
                                   placeholder="https://www.youtube.com/watch?v=UemFRPrl1hk" 
                                   value="https://www.youtube.com/watch?v=UemFRPrl1hk">
                        </div>
                        <div class="col-lg-2">
                            <label class="eyebrow-label mb-1">Timestamp (Sec)</label>
                            <input type="number" id="yt-timestamp-input" class="form-control auth-input" min="0" step="1" value="5">
                        </div>
                        <div class="col-lg-2 d-flex align-items-end">
                            <button class="btn-void-violet w-100" id="btn-yt-inspect" onclick="inspectYouTubeVideo()">
                                <i class="fas fa-play me-1"></i> Inspect
                            </button>
                        </div>
                    </div>

                    <!-- Quick Preset Buttons -->
                    <div class="d-flex align-items-center gap-2 mb-3">
                        <span class="eyebrow-label">Presets:</span>
                        <button class="btn-ghost-pill btn-sm" onclick="setYtPreset('https://www.youtube.com/watch?v=UemFRPrl1hk', 5)">
                            <i class="fab fa-youtube text-danger me-1"></i> Road Surveillance (UemFRPrl1hk)
                        </button>
                    </div>

                    <!-- YouTube Video Meta Info -->
                    <div id="yt-info-card" class="steel-subplate p-3 mb-3" style="display: none;">
                        <div class="d-flex justify-content-between align-items-center">
                            <div>
                                <span class="eyebrow-label">STREAM METADATA</span>
                                <div class="fw-semibold text-truncate" id="yt-meta-title" style="color: var(--color-text-primary); max-width: 650px;">--</div>
                            </div>
                            <span class="auth-badge" id="yt-meta-time">0s</span>
                        </div>
                    </div>
                </div>
            </div>

            <!-- ------------------------------------------------------------------
                 TAB 3: VIDEO FILE UPLOAD ANALYZER
                 ------------------------------------------------------------------ -->
            <div class="tab-pane fade" id="tab-video" role="tabpanel">
                <div class="glass-plate p-4">
                    <span class="eyebrow-label"><i class="fas fa-film me-1"></i> Local Video File Inspector</span>
                    <h5 class="fw-bold mb-2" style="font-family: var(--font-aeonikpro); color: var(--color-text-primary);">Upload MP4 / AVI Traffic Video</h5>
                    <p class="small mb-4" style="color: var(--color-text-muted);">
                        Upload any traffic CCTV recording (.mp4, .avi, .mov) and choose a timestamp frame to detect helmet violations.
                    </p>

                    <div class="row g-3 align-items-end mb-3">
                        <div class="col-lg-7">
                            <label class="eyebrow-label mb-1">Select Video File</label>
                            <input type="file" id="video-file-input" class="form-control auth-input" accept="video/*">
                        </div>
                        <div class="col-lg-3">
                            <label class="eyebrow-label mb-1">Target Second (Timestamp)</label>
                            <input type="number" id="video-sec-input" class="form-control auth-input" min="0" step="1" value="0">
                        </div>
                        <div class="col-lg-2">
                            <button class="btn-void-violet w-100" onclick="uploadAndInspectVideo()">
                                <i class="fas fa-bolt me-1"></i> Scan Frame
                            </button>
                        </div>
                    </div>
                </div>
            </div>

            <!-- ------------------------------------------------------------------
                 TAB 4: LIVE WEBCAM HUD
                 ------------------------------------------------------------------ -->
            <div class="tab-pane fade" id="tab-webcam" role="tabpanel">
                <div class="glass-plate p-4 text-center">
                    <span class="eyebrow-label"><i class="fas fa-video me-1"></i> Real-Time Surveillance</span>
                    <h5 class="fw-bold mb-2" style="font-family: var(--font-aeonikpro); color: var(--color-text-primary);">Live Browser WebCam Violation Stream</h5>
                    <p class="small mb-3" style="color: var(--color-text-muted);">Streams live frames from your connected camera to the YOLOv8 violation engine in real time.</p>

                    <div class="d-flex justify-content-center gap-3 mb-3">
                        <button class="btn-void-violet px-4" id="btn-start-cam" onclick="startWebcam()">
                            <i class="fas fa-play me-2"></i> Start Camera Stream
                        </button>
                        <button class="btn-ghost-pill px-4" id="btn-stop-cam" onclick="stopWebcam()" disabled>
                            <i class="fas fa-stop me-2"></i> Terminate Stream
                        </button>
                    </div>

                    <div class="row justify-content-center">
                        <div class="col-lg-8">
                            <div class="viewport-frame position-relative" style="min-height: 420px;">
                                <video id="webcam-video" autoplay playsinline muted style="display: none;"></video>
                                <canvas id="webcam-canvas" style="display: none;"></canvas>
                                <img id="webcam-output" src="" alt="Live Stream" style="width: 100%; display: none;">
                                <div id="cam-placeholder" class="py-5 text-light opacity-75">
                                    <i class="fas fa-camera fa-3x mb-3 text-secondary"></i>
                                    <h6>Camera is offline</h6>
                                    <p class="small mb-0 text-secondary">Click "Start Camera Stream" to launch real-time video violation monitoring.</p>
                                </div>
                            </div>
                            <div class="mt-3 d-flex justify-content-around small font-monospace" style="color: var(--color-text-secondary);">
                                <span>STREAM FPS: <b id="cam-fps" style="color: var(--color-void-violet);">0</b></span>
                                <span>SAFE RIDERS: <b id="cam-safe" style="color: var(--color-deep-teal);">0</b></span>
                                <span>VIOLATIONS: <b id="cam-viol" style="color: var(--color-ember-glow);">0</b></span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- ------------------------------------------------------------------
                 TAB 5: DIRECT IMAGE URL STREAM
                 ------------------------------------------------------------------ -->
            <div class="tab-pane fade" id="tab-url" role="tabpanel">
                <div class="glass-plate p-4">
                    <span class="eyebrow-label"><i class="fas fa-link me-1"></i> Direct Network Source</span>
                    <h5 class="fw-bold mb-3" style="font-family: var(--font-aeonikpro); color: var(--color-text-primary);">Analyze Online Traffic Camera Image URL</h5>
                    
                    <div class="input-group mb-3">
                        <input type="text" id="url-input" class="form-control auth-input" placeholder="Paste direct image URL (e.g. https://example.com/traffic_scene.jpg)...">
                        <button class="btn-void-violet px-4" onclick="analyzeUrl()">Analyze Link</button>
                    </div>
                    <div class="small" style="color: var(--color-text-muted);">
                        <i class="fas fa-info-circle me-1"></i> Downloads the remote camera image directly into server RAM and processes it through the YOLOv8 pipeline.
                    </div>
                </div>
            </div>

            <!-- ------------------------------------------------------------------
                 TAB 6: INCIDENT LOG & ANALYTICS
                 ------------------------------------------------------------------ -->
            <div class="tab-pane fade" id="tab-history" role="tabpanel">
                <div class="glass-plate p-4">
                    <div class="d-flex justify-content-between align-items-center mb-3">
                        <div>
                            <span class="eyebrow-label"><i class="fas fa-chart-line me-1"></i> Incident Audit Trail</span>
                            <h5 class="fw-bold mb-0" style="font-family: var(--font-aeonikpro); color: var(--color-text-primary);">Session Scans & Telemetry Log</h5>
                        </div>
                        <div class="d-flex gap-2">
                            <button class="btn-ghost-pill btn-sm" onclick="exportHistoryCSV()"><i class="fas fa-file-csv me-1"></i> Export CSV</button>
                            <button class="btn-ghost-pill btn-sm" onclick="clearHistory()"><i class="fas fa-trash me-1"></i> Clear Log</button>
                        </div>
                    </div>

                    <div class="table-responsive">
                        <table class="table cathedral-table">
                            <thead>
                                <tr>
                                    <th>ID</th>
                                    <th>Source</th>
                                    <th>Timestamp</th>
                                    <th>Thumbnail</th>
                                    <th>Safe Riders</th>
                                    <th>Violations</th>
                                    <th>Compliance</th>
                                    <th>Status</th>
                                    <th>Latency</th>
                                </tr>
                            </thead>
                            <tbody id="history-tbody">
                                <tr>
                                    <td colspan="9" class="text-center py-4" style="color: var(--color-text-muted);">No scans recorded yet in this session.</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>

        </div> <!-- /tab-content -->
    </div> <!-- /container -->

    <!-- AUDIO CHIME FOR VIOLATION ALERT -->
    <audio id="alert-audio" preload="auto">
        <source src="https://assets.mixkit.co/active_storage/sfx/2869/2869-preview.mp3" type="audio/mpeg">
    </audio>

    <!-- Bootstrap 5 JS -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>

    <!-- CLIENT LOGIC SCRIPT -->
    <script>
        let currentResult = null;
        let currentViewMode = 'annotated';
        let currentFile = null;
        let audioEnabled = true;
        let isWebcamRunning = false;
        let webcamStream = null;
        let webcamInterval = null;

        // Initialize on DOM load
        document.addEventListener('DOMContentLoaded', () => {
            fetchSystemInfo();
            loadSampleGallery();
            setupDragAndDrop();
            setupClipboardPaste();
            initTheme();
        });

        // Theme Toggle (White Light Mode vs Dark Mode)
        function initTheme() {
            const savedTheme = localStorage.getItem('authguard-theme') || 'light';
            setTheme(savedTheme);
        }

        function toggleTheme() {
            const current = document.documentElement.getAttribute('data-theme') || 'light';
            const next = (current === 'light') ? 'dark' : 'light';
            setTheme(next);
        }

        function setTheme(theme) {
            document.documentElement.setAttribute('data-theme', theme);
            localStorage.setItem('authguard-theme', theme);
            const btn = document.getElementById('btn-theme');
            if (btn) {
                if (theme === 'light') {
                    btn.innerHTML = '<i class="fas fa-moon me-1"></i> Dark Mode';
                } else {
                    btn.innerHTML = '<i class="fas fa-sun me-1"></i> Light Mode';
                }
            }
        }

        // 1. Fetch System Metadata
        async function fetchSystemInfo() {
            try {
                const res = await fetch('/system_info');
                const data = await res.json();
                document.getElementById('nav-device').innerText = data.device_name;
            } catch(e) {
                document.getElementById('nav-device').innerText = 'Online';
            }
        }

        // 2. Drag & Drop and Clipboard Setup
        function setupDragAndDrop() {
            const dropZone = document.getElementById('drop-zone');
            ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(evt => {
                dropZone.addEventListener(evt, e => { e.preventDefault(); e.stopPropagation(); });
            });
            dropZone.addEventListener('dragover', () => dropZone.classList.add('dragover'));
            dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
            dropZone.addEventListener('drop', e => {
                dropZone.classList.remove('dragover');
                const file = e.dataTransfer.files[0];
                if (file) handleFileUpload(file);
            });
        }

        function setupClipboardPaste() {
            window.addEventListener('paste', e => {
                const items = (e.clipboardData || e.originalEvent.clipboardData).items;
                for (let item of items) {
                    if (item.type.indexOf('image') !== -1) {
                        const file = item.getAsFile();
                        handleFileUpload(file);
                        break;
                    }
                }
            });
        }

        // 3. File Upload Handler
        async function handleFileUpload(file) {
            if (!file) return;
            currentFile = file;
            const conf = document.getElementById('conf-slider').value / 100.0;
            const iou = document.getElementById('iou-slider').value / 100.0;

            const formData = new FormData();
            formData.append('file', file);
            formData.append('conf', conf);
            formData.append('iou', iou);

            document.getElementById('tab-image-btn').click();
            await sendInferenceRequest('/predict', formData, false);
        }

        // 4. Sample Images Gallery
        async function loadSampleGallery() {
            const container = document.getElementById('sample-gallery-container');
            container.innerHTML = '<div class="small text-center py-2" style="color: var(--color-text-muted);"><i class="fas fa-spinner fa-spin me-1"></i> Loading samples...</div>';
            try {
                const res = await fetch('/sample_images');
                const data = await res.json();
                if (!data.samples || data.samples.length === 0) {
                    container.innerHTML = '<div class="small text-center py-2" style="color: var(--color-text-muted);">No local samples found.</div>';
                    return;
                }
                container.innerHTML = '';
                data.samples.slice(0, 6).forEach(sample => {
                    const col = document.createElement('div');
                    col.className = 'col-4';
                    col.innerHTML = `
                        <div class="sample-tile" onclick="predictSampleByName('${sample.filename}')">
                            <img src="${sample.thumbnail}" alt="${sample.filename}">
                            <div class="p-1 text-truncate font-monospace" style="font-size: 0.62rem; color: var(--color-text-muted);">${sample.filename}</div>
                        </div>
                    `;
                    container.appendChild(col);
                });
            } catch(e) {
                container.innerHTML = '<div class="small text-center py-2" style="color: var(--color-text-muted);">Failed to load samples.</div>';
            }
        }

        async function predictSampleByName(filename) {
            const conf = document.getElementById('conf-slider').value / 100.0;
            const iou = document.getElementById('iou-slider').value / 100.0;
            const payload = { filename, conf, iou };
            document.getElementById('tab-image-btn').click();
            await sendInferenceRequest('/predict_sample', payload, true);
        }

        // 5. YouTube Video Stream Inspector
        function setYtPreset(url, sec) {
            document.getElementById('yt-url-input').value = url;
            document.getElementById('yt-timestamp-input').value = sec;
        }

        async function inspectYouTubeVideo() {
            const url = document.getElementById('yt-url-input').value.trim();
            const timestamp = parseFloat(document.getElementById('yt-timestamp-input').value) || 0.0;
            const conf = document.getElementById('conf-slider').value / 100.0;
            const iou = document.getElementById('iou-slider').value / 100.0;

            if (!url) {
                alert('Please enter a valid YouTube video URL');
                return;
            }

            const btn = document.getElementById('btn-yt-inspect');
            btn.disabled = true;
            btn.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i> Extracting Frame...';

            try {
                const res = await fetch('/predict_youtube', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ url, timestamp, conf, iou })
                });
                const data = await res.json();
                if (data.error) {
                    alert('YouTube Error: ' + data.error);
                    return;
                }

                // Show metadata card
                if (data.youtube_info) {
                    document.getElementById('yt-info-card').style.display = 'block';
                    document.getElementById('yt-meta-title').innerText = data.youtube_info.title;
                    document.getElementById('yt-meta-time').innerText = `At ${timestamp}s / ${data.youtube_info.duration}s`;
                }

                // Switch to Image Inspector tab to view results
                document.getElementById('tab-image-btn').click();
                currentResult = data;
                renderDetectionResults(data);

            } catch(e) {
                alert('YouTube inspection failed: ' + e.message);
            } finally {
                btn.disabled = false;
                btn.innerHTML = '<i class="fas fa-play me-1"></i> Inspect';
            }
        }

        // 6. Video File Upload
        async function uploadAndInspectVideo() {
            const fileInput = document.getElementById('video-file-input');
            const file = fileInput.files[0];
            const timestamp = parseFloat(document.getElementById('video-sec-input').value) || 0.0;
            const conf = document.getElementById('conf-slider').value / 100.0;
            const iou = document.getElementById('iou-slider').value / 100.0;

            if (!file) {
                alert('Please select a video file (.mp4, .avi, .mov)');
                return;
            }

            const formData = new FormData();
            formData.append('file', file);
            formData.append('timestamp_sec', timestamp);
            formData.append('conf', conf);
            formData.append('iou', iou);

            document.getElementById('tab-image-btn').click();
            await sendInferenceRequest('/predict_video_upload', formData, false);
        }

        // 7. Image URL Analyzer
        async function analyzeUrl() {
            const url = document.getElementById('url-input').value.trim();
            if (!url) { alert('Please enter a valid image URL'); return; }
            const conf = document.getElementById('conf-slider').value / 100.0;
            const iou = document.getElementById('iou-slider').value / 100.0;
            document.getElementById('tab-image-btn').click();
            await sendInferenceRequest('/predict_url', { url, conf, iou }, true);
        }

        // 8. Sliders
        function updateConfidence(val) {
            document.getElementById('conf-val').innerText = val + '%';
            if (currentFile) {
                handleFileUpload(currentFile);
            }
        }

        // 9. Inference Requester
        async function sendInferenceRequest(endpoint, payload, isJson) {
            const loader = document.getElementById('loader');
            const placeholder = document.getElementById('viewport-placeholder');
            const imgEl = document.getElementById('viewport-img');
            const sideContainer = document.getElementById('side-by-side-container');

            loader.style.display = 'block';
            placeholder.style.display = 'none';
            imgEl.style.display = 'none';
            sideContainer.style.display = 'none';

            try {
                const options = {
                    method: 'POST',
                    headers: isJson ? { 'Content-Type': 'application/json' } : {},
                    body: isJson ? JSON.stringify(payload) : payload
                };
                const res = await fetch(endpoint, options);
                const data = await res.json();

                if (data.error) {
                    alert('Error: ' + data.error);
                    placeholder.style.display = 'block';
                    return;
                }

                currentResult = data;
                renderDetectionResults(data);

            } catch(err) {
                alert('Inference request failed: ' + err.message);
                placeholder.style.display = 'block';
            } finally {
                loader.style.display = 'none';
            }
        }

        // 10. Render Results
        function renderDetectionResults(data) {
            document.getElementById('stat-safe').innerText = data.with_helmet;
            document.getElementById('stat-violation').innerText = data.without_helmet;
            document.getElementById('stat-compliance').innerText = data.compliance_rate + '%';
            document.getElementById('stat-latency').innerHTML = data.latency_ms + '<small style="font-size:0.75rem">ms</small>';

            const pill = document.getElementById('status-pill');
            if (data.violation) {
                pill.className = 'cathedral-status-pill violation';
                pill.innerHTML = `<i class="fas fa-triangle-exclamation"></i> ${data.without_helmet} VIOLATION(S) DETECTED`;
                if (audioEnabled) playAlertChime();
            } else {
                pill.className = 'cathedral-status-pill safe';
                pill.innerHTML = `<i class="fas fa-shield-check"></i> 100% COMPLIANT &bull; ALL RIDERS SAFE`;
            }

            // Image display update
            updateViewportImage();

            // Populate Breakdown Table
            const tbody = document.getElementById('detections-tbody');
            tbody.innerHTML = '';
            document.getElementById('objects-count-badge').innerText = `${data.detections.length} objects`;

            if (data.detections.length === 0) {
                tbody.innerHTML = '<tr><td colspan="5" class="text-center py-2" style="color: var(--color-text-muted);">No riders detected at current threshold.</td></tr>';
            } else {
                data.detections.forEach(d => {
                    const tr = document.createElement('tr');
                    const isViol = d.is_violation;
                    const statusColor = isViol ? 'color: var(--color-ember-glow);' : 'color: var(--color-deep-teal);';
                    const badgeBg = isViol ? 'background: rgba(220, 38, 38, 0.1); border: 1px solid var(--color-ember-glow); color: var(--color-ember-glow);' : 'background: rgba(5, 150, 105, 0.1); border: 1px solid var(--color-deep-teal); color: var(--color-deep-teal);';
                    
                    tr.innerHTML = `
                        <td class="font-monospace" style="color: var(--color-text-muted);">${d.id}</td>
                        <td class="fw-semibold" style="${statusColor}">${d.class}</td>
                        <td>
                            <div class="d-flex align-items-center gap-2">
                                <span class="font-monospace">${d.confidence}%</span>
                                <div class="progress flex-grow-1" style="height: 5px; background: rgba(0,0,0,0.06);">
                                    <div class="progress-bar" style="width: ${d.confidence}%; background: ${isViol ? 'var(--color-ember-glow)' : 'var(--color-deep-teal)'};"></div>
                                </div>
                            </div>
                        </td>
                        <td class="font-monospace small" style="color: var(--color-text-muted);">[${d.box.join(', ')}]</td>
                        <td><span class="auth-badge" style="${badgeBg}">${isViol ? 'VIOLATION' : 'SAFE'}</span></td>
                    `;
                    tbody.appendChild(tr);
                });
            }
            document.getElementById('table-container').style.display = 'block';
        }

        // 11. View Modes
        function switchView(mode) {
            currentViewMode = mode;
            ['annotated', 'side', 'original'].forEach(m => {
                document.getElementById('btn-view-' + m).classList.toggle('active', m === mode);
            });
            updateViewportImage();
        }

        function updateViewportImage() {
            if (!currentResult) return;
            const imgEl = document.getElementById('viewport-img');
            const sideContainer = document.getElementById('side-by-side-container');

            if (currentViewMode === 'side') {
                imgEl.style.display = 'none';
                sideContainer.style.display = 'flex';
                document.getElementById('side-img-orig').src = currentResult.original_url;
                document.getElementById('side-img-ann').src = currentResult.image_url;
            } else {
                sideContainer.style.display = 'none';
                imgEl.style.display = 'block';
                imgEl.src = (currentViewMode === 'original') ? currentResult.original_url : currentResult.image_url;
            }
        }

        // 12. Audio Alerts
        function toggleAudioAlert() {
            audioEnabled = !audioEnabled;
            const btn = document.getElementById('btn-audio');
            if (audioEnabled) {
                btn.innerHTML = '<i class="fas fa-volume-high me-1"></i> Sound: ON';
                btn.classList.remove('active');
            } else {
                btn.innerHTML = '<i class="fas fa-volume-xmark me-1"></i> Sound: OFF';
                btn.classList.add('active');
            }
        }

        function playAlertChime() {
            try {
                const audio = document.getElementById('alert-audio');
                audio.currentTime = 0;
                audio.play().catch(e => {});
            } catch(e) {}
        }

        // 13. Download & Export
        function downloadAnnotatedImage() {
            if (!currentResult || !currentResult.image_url) { alert('No image available to download'); return; }
            const a = document.createElement('a');
            a.href = currentResult.image_url;
            a.download = `traffic_guard_${Date.now()}.jpg`;
            a.click();
        }

        function copyDetectionsJSON() {
            if (!currentResult) { alert('No detection report available'); return; }
            navigator.clipboard.writeText(JSON.stringify(currentResult, null, 2));
            alert('JSON Telemetry report copied to clipboard!');
        }

        // 14. History Tab
        async function loadHistoryTable() {
            const tbody = document.getElementById('history-tbody');
            try {
                const res = await fetch('/detection_history');
                const data = await res.json();
                if (!data.history || data.history.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="9" class="text-center py-4" style="color: var(--color-text-muted);">No scans recorded yet in this session.</td></tr>';
                    return;
                }
                tbody.innerHTML = '';
                data.history.forEach(item => {
                    const tr = document.createElement('tr');
                    const isViol = item.violations > 0;
                    const badgeBg = isViol ? 'background: rgba(220, 38, 38, 0.1); border: 1px solid var(--color-ember-glow); color: var(--color-ember-glow);' : 'background: rgba(5, 150, 105, 0.1); border: 1px solid var(--color-deep-teal); color: var(--color-deep-teal);';
                    tr.innerHTML = `
                        <td class="font-monospace" style="color: var(--color-text-muted);">${item.id}</td>
                        <td class="eyebrow-label">${item.source || 'Scan'}</td>
                        <td class="font-monospace small" style="color: var(--color-text-muted);">${item.timestamp}</td>
                        <td><img src="${item.thumbnail}" style="width: 48px; height: 34px; object-fit: cover; border-radius: 4px; border: 1px solid var(--color-border);"></td>
                        <td class="fw-semibold" style="color: var(--color-deep-teal);">${item.safe}</td>
                        <td class="fw-semibold" style="color: var(--color-ember-glow);">${item.violations}</td>
                        <td>${item.compliance}%</td>
                        <td><span class="auth-badge" style="${badgeBg}">${item.status}</span></td>
                        <td class="font-monospace small" style="color: var(--color-text-muted);">${item.latency_ms} ms</td>
                    `;
                    tbody.appendChild(tr);
                });
            } catch(e) {}
        }

        function exportHistoryCSV() {
            window.location.href = '/export_history_csv';
        }

        async function clearHistory() {
            await fetch('/clear_history', { method: 'POST' });
            loadHistoryTable();
        }

        // 15. Live WebCam Streaming
        async function startWebcam() {
            const video = document.getElementById('webcam-video');
            const canvas = document.getElementById('webcam-canvas');
            const output = document.getElementById('webcam-output');
            const placeholder = document.getElementById('cam-placeholder');

            try {
                webcamStream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
                video.srcObject = webcamStream;
                video.play();

                placeholder.style.display = 'none';
                output.style.display = 'block';

                document.getElementById('btn-start-cam').disabled = true;
                document.getElementById('btn-stop-cam').disabled = false;
                isWebcamRunning = true;

                let lastTime = performance.now();
                let frameCount = 0;

                webcamInterval = setInterval(async () => {
                    if (!isWebcamRunning) return;

                    canvas.width = video.videoWidth || 640;
                    canvas.height = video.videoHeight || 480;
                    const ctx = canvas.getContext('2d');
                    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

                    const dataUrl = canvas.toDataURL('image/jpeg', 0.65);
                    const conf = document.getElementById('conf-slider').value / 100.0;

                    try {
                        const res = await fetch('/detect_webcam_frame', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ frame: dataUrl, conf })
                        });
                        const result = await res.json();
                        if (result.image_url) {
                            output.src = result.image_url;
                            document.getElementById('cam-safe').innerText = result.with_helmet;
                            document.getElementById('cam-viol').innerText = result.without_helmet;
                        }

                        // FPS Calculation
                        frameCount++;
                        const now = performance.now();
                        if (now - lastTime >= 1000) {
                            document.getElementById('cam-fps').innerText = frameCount;
                            frameCount = 0;
                            lastTime = now;
                        }
                    } catch(e) {}
                }, 200);

            } catch(e) {
                alert('Could not access webcam: ' + e.message);
            }
        }

        function stopWebcam() {
            isWebcamRunning = false;
            if (webcamInterval) clearInterval(webcamInterval);
            if (webcamStream) {
                webcamStream.getTracks().forEach(t => t.stop());
            }
            document.getElementById('btn-start-cam').disabled = false;
            document.getElementById('btn-stop-cam').disabled = true;
            document.getElementById('webcam-output').style.display = 'none';
            document.getElementById('cam-placeholder').style.display = 'block';
        }
    </script>
</body>
</html>
"""


# ======================================================================================
# API ENDPOINTS
# ======================================================================================

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serves the White Cathedral Edition UI."""
    return HTML_CONTENT


@app.get("/system_info")
async def system_info():
    """Returns runtime hardware and model metadata."""
    return {
        "device_name": f"{DEVICE_NAME} Acceleration",
        "device_type": str(DEVICE),
        "model_loaded": model is not None,
        "classes": list(model.names.values()) if model else ["With Helmet", "Without Helmet"]
    }


@app.get("/sample_images")
async def get_sample_images():
    """Returns sample images with thumbnails from local dataset."""
    found_images = []
    for d in SAMPLE_DIRS:
        if d.exists():
            for p in list(d.glob("*.png")) + list(d.glob("*.jpg")) + list(d.glob("*.jpeg")):
                found_images.append(p)

    if not found_images:
        return {"samples": []}

    # Pick 8 diverse samples
    selected = found_images[:8]
    samples_data = []

    for img_path in selected:
        img = cv2.imread(str(img_path))
        if img is not None:
            thumb = cv2.resize(img, (160, 110))
            _, buf = cv2.imencode('.jpg', thumb)
            b64_thumb = base64.b64encode(buf).decode('utf-8')
            samples_data.append({
                "filename": img_path.name,
                "thumbnail": f"data:image/jpeg;base64,{b64_thumb}"
            })

    return {"samples": samples_data}


@app.post("/predict")
async def predict_upload(
    file: UploadFile = File(...),
    conf: float = Form(0.25),
    iou: float = Form(0.45)
):
    """Processes uploaded image file."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image file format."})

    return run_pipeline(img, conf=conf, iou=iou, source_label="Upload")


@app.post("/predict_sample")
async def predict_sample(payload: dict = Body(...)):
    """Runs prediction on a sample file from disk."""
    filename = payload.get("filename")
    conf = float(payload.get("conf", 0.25))
    iou = float(payload.get("iou", 0.45))

    target_path = None
    for d in SAMPLE_DIRS:
        p = d / filename
        if p.exists():
            target_path = p
            break

    if not target_path:
        return JSONResponse(status_code=404, content={"error": f"Sample image {filename} not found."})

    img = cv2.imread(str(target_path))
    if img is None:
        return JSONResponse(status_code=400, content={"error": "Could not decode sample image."})

    return run_pipeline(img, conf=conf, iou=iou, source_label=f"Sample ({filename})")


@app.post("/predict_url")
async def predict_url(payload: dict = Body(...)):
    """Downloads image from URL and executes detection."""
    url = payload.get("url")
    conf = float(payload.get("conf", 0.25))
    iou = float(payload.get("iou", 0.45))

    if not url:
        return JSONResponse(status_code=400, content={"error": "No URL provided."})

    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as resp:
            arr = np.asarray(bytearray(resp.read()), dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                return JSONResponse(status_code=400, content={"error": "Could not decode image from URL."})
            return run_pipeline(img, conf=conf, iou=iou, source_label="Image URL")
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Failed to download image: {str(e)}"})


@app.post("/predict_youtube")
async def predict_youtube(payload: dict = Body(...)):
    """Extracts a frame from YouTube video URL and executes YOLOv8 detection."""
    url = payload.get("url", "").strip()
    timestamp_sec = float(payload.get("timestamp", 0.0))
    conf = float(payload.get("conf", 0.25))
    iou = float(payload.get("iou", 0.45))

    if not url:
        return JSONResponse(status_code=400, content={"error": "No YouTube URL provided."})

    try:
        frame, title, duration = extract_youtube_frame(url, timestamp_sec=timestamp_sec)
        resp = run_pipeline(frame, conf=conf, iou=iou, source_label="YouTube Stream")
        resp["youtube_info"] = {
            "title": title,
            "duration": duration,
            "timestamp": timestamp_sec,
            "url": url
        }
        return resp
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"YouTube extraction failed: {str(e)}"})


@app.post("/predict_video_upload")
async def predict_video_upload(
    file: UploadFile = File(...),
    timestamp_sec: float = Form(0.0),
    conf: float = Form(0.25),
    iou: float = Form(0.45)
):
    """Extracts frame from uploaded video at specified timestamp and runs inference."""
    temp_path = PROJECT_ROOT / f"temp_upload_{int(time.time() * 1000)}.mp4"
    try:
        contents = await file.read()
        with open(temp_path, "wb") as f:
            f.write(contents)

        cap = cv2.VideoCapture(str(temp_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = total_frames / fps if fps > 0 else 0

        if timestamp_sec > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(timestamp_sec) * 1000.0)

        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            return JSONResponse(status_code=400, content={"error": "Could not read frame from uploaded video."})

        resp = run_pipeline(frame, conf=conf, iou=iou, source_label=f"Video ({file.filename})")
        resp["video_info"] = {
            "filename": file.filename,
            "duration": round(duration, 1),
            "timestamp": round(timestamp_sec, 1)
        }
        return resp
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass


@app.post("/detect_webcam_frame")
async def detect_webcam_frame(payload: dict = Body(...)):
    """Fast detection endpoint for base64 browser webcam stream."""
    frame_b64 = payload.get("frame")
    conf = float(payload.get("conf", 0.25))

    if not frame_b64:
        return {"error": "No frame received"}

    if "," in frame_b64:
        frame_b64 = frame_b64.split(",")[1]

    raw_bytes = base64.b64decode(frame_b64)
    nparr = np.frombuffer(raw_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None or model is None:
        return {"error": "Decode error"}

    results = model.predict(img, conf=conf, verbose=False)[0]
    annotated, stats, _ = annotate_frame(img, [results], conf_threshold=conf)

    _, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
    b64_out = base64.b64encode(buf).decode('utf-8')

    return {
        "with_helmet": stats.get("With Helmet", 0),
        "without_helmet": stats.get("Without Helmet", 0),
        "image_url": f"data:image/jpeg;base64,{b64_out}"
    }


@app.get("/detection_history")
async def get_history():
    """Returns the session history log."""
    return {"history": DETECTION_HISTORY}


@app.post("/clear_history")
async def clear_history():
    """Clears session history log."""
    global DETECTION_HISTORY
    DETECTION_HISTORY = []
    return {"success": True}


@app.get("/export_history_csv")
async def export_history_csv():
    """Exports session incident log as CSV."""
    lines = ["ID,Source,Timestamp,Safe_Count,Violations_Count,Total_Riders,Compliance_Percent,Status,Latency_ms"]
    for h in DETECTION_HISTORY:
        lines.append(f"{h['id']},{h.get('source', 'Scan')},{h['timestamp']},{h['safe']},{h['violations']},{h['total']},{h['compliance']},{h['status']},{h['latency_ms']}")

    csv_data = "\n".join(lines)
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=traffic_incident_history_{int(time.time())}.csv"}
    )


# ======================================================================================
# APPLICATION ENTRYPOINT
# ======================================================================================

if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("🚦 AI TRAFFIC GUARD — WHITE CATHEDRAL EDITION LAUNCHER")
    print("=" * 65)
    print(f"Device    : {DEVICE_NAME}")
    print(f"Weights   : {MODEL_PATH}")
    print(f"Server URL: http://127.0.0.1:8000")
    print("=" * 65 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
