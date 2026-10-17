import os
import cv2
import numpy as np
import base64
import time
from datetime import datetime
from pathlib import Path
import torch
from ultralytics import YOLO
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

# Fix OpenMP duplicate library conflict on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- CONFIGURATION ---
PROJECT_ROOT = Path(__file__).resolve().parent
DEVICE = 0 if torch.cuda.is_available() else "cpu"
DEVICE_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

# Model Path - Use your best trained model
MODEL_PATH = PROJECT_ROOT / "saved_models" / "best.pt"
if not MODEL_PATH.exists():
    # Fallback to other common paths if best.pt is not in saved_models
    alternatives = [
        PROJECT_ROOT / "runs" / "detect" / "train" / "weights" / "best.pt",
        PROJECT_ROOT / "yolov8n.pt"
    ]
    for alt in alternatives:
        if alt.exists():
            MODEL_PATH = alt
            break

# Load Model
model = YOLO(str(MODEL_PATH))
print(f"🚀 Server Loaded Model: {MODEL_PATH} on {DEVICE_NAME}")

app = FastAPI()

# Color mappings (BGR)
COLOR_SAFE = (105, 150, 5)      # Deep Teal
COLOR_VIOLATION = (38, 38, 220) # Vivid Red
COLOR_NEUTRAL = (246, 130, 59)  # Blueprint Blue

def classify_detection(cls_name: str):
    name_clean = str(cls_name).lower().replace("-", " ").replace("_", " ").strip()
    if any(k in name_clean for k in ["without helmet", "no helmet", "violat", "barehead", "head"]):
        return "violation", True, "VIOLATION"
    if any(k in name_clean for k in ["with helmet", "helmet", "safe", "compliant"]):
        return "safe", False, "SAFE"
    return "neutral", False, "OBJECT"

def annotate_frame(image_bgr, results, conf_threshold=0.25):
    annotated = image_bgr.copy()
    h, w = annotated.shape[:2]
    stats = {"With Helmet": 0, "Without Helmet": 0, "Other": 0}
    detections = []

    for r in results:
        boxes = getattr(r, "boxes", None)
        if boxes is not None:
            for box in boxes:
                conf = float(box.conf[0])
                if conf < conf_threshold: continue

                cls_id = int(box.cls[0])
                cls_name = r.names.get(cls_id, str(cls_id))
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                det_type, is_violation, status_text = classify_detection(cls_name)
                color = COLOR_VIOLATION if det_type == "violation" else (COLOR_SAFE if det_type == "safe" else COLOR_NEUTRAL)

                if det_type == "violation": stats["Without Helmet"] += 1
                elif det_type == "safe": stats["With Helmet"] += 1
                else: stats["Other"] += 1

                detections.append({
                    "class": cls_name,
                    "confidence": round(conf * 100, 1),
                    "is_violation": is_violation,
                    "box": [x1, y1, x2, y2]
                })

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, f"{cls_name} {conf*100:.1f}%", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return annotated, stats, detections

@app.get("/", response_class=HTMLResponse)
async def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Traffic Violation Server</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body { background: #f8fafc; padding: 50px; font-family: sans-serif; }
            .upload-card { background: white; border-radius: 15px; padding: 30px; box-shadow: 0 4px 20px rgba(0,0,0,0.05); max-width: 800px; margin: auto; text-align: center; }
            #result-img { max-width: 100%; border-radius: 10px; margin-top: 20px; display: none; }
            .status-badge { font-size: 1.2rem; padding: 10px 20px; border-radius: 50px; display: inline-block; margin-top: 15px; display: none; }
        </style>
    </head>
    <body>
        <div class="upload-card">
            <h2>🚦 Traffic Violation Detector</h2>
            <p class="text-muted">Upload an image to check for helmet compliance</p>
            <div class="mb-3">
                <input type="file" id="fileInput" class="form-control" accept="image/*">
            </div>
            <button onclick="uploadImage()" class="btn btn-primary btn-lg w-100">Analyze Image</button>
            <div id="status" class="status-badge"></div>
            <img id="result-img">
        </div>

        <script>
            async function uploadImage() {
                const fileInput = document.getElementById('fileInput');
                if (!fileInput.files[0]) return alert('Please select an image');

                const formData = new FormData();
                formData.append('file', fileInput.files[0]);

                const imgEl = document.getElementById('result-img');
                const statusEl = document.getElementById('status');

                imgEl.style.display = 'none';
                statusEl.style.display = 'none';

                const res = await fetch('/predict', { method: 'POST', body: formData });
                const data = await res.json();

                if (data.success) {
                    imgEl.src = data.image_url;
                    imgEl.style.display = 'block';
                    statusEl.style.display = 'inline-block';
                    statusEl.innerText = data.violation ? '⚠️ VIOLATION DETECTED' : '✅ COMPLIANT';
                    statusEl.className = 'status-badge ' + (data.violation ? 'bg-danger text-white' : 'bg-success text-white');
                } else {
                    alert('Error: ' + data.error);
                }
            }
        </script>
    </body>
    </html>
    """

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return JSONResponse(status_code=400, content={"success": False, "error": "Invalid image file"})

        results = model.predict(img, conf=0.25, device=DEVICE, verbose=False)[0]
        annotated, stats, detections = annotate_frame(img, [results])

        _, buf = cv2.imencode('.jpg', annotated)
        b64_img = base64.b64encode(buf).decode('utf-8')

        viol_count = stats["Without Helmet"]

        return {
            "success": True,
            "violation": viol_count > 0,
            "with_helmet": stats["With Helmet"],
            "without_helmet": viol_count,
            "image_url": f"data:image/jpeg;base64,{b64_img}",
            "detections": detections
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
