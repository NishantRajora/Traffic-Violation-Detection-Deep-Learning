import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path

from flask import Flask, request, jsonify

from ultralytics import YOLO

from PIL import Image


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

MODEL_PATH = Path("saved_models/helmet_model.pt")


if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found at: {MODEL_PATH.resolve()}"
    )


model = YOLO(str(MODEL_PATH))


print("=" * 60)
print("HELMET DETECTION WEB APP")
print("=" * 60)

print("Model loaded successfully!")
print("Model path:", MODEL_PATH.resolve())
print("Classes:", model.names)

print("=" * 60)


# ============================================================
# HTML PAGE
# ============================================================

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">

<head>

    <meta charset="UTF-8">

    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <title>Helmet AI | Traffic Violation Detection</title>

    <style>
        /* ============================================================
   RESET
============================================================ */

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }


        body {

            font-family:
                Inter,
                -apple-system,
                BlinkMacSystemFont,
                "Segoe UI",
                sans-serif;

            min-height: 100vh;

            background:
                radial-gradient(circle at 10% 10%,
                    rgba(59, 130, 246, 0.15),
                    transparent 30%),
                radial-gradient(circle at 90% 90%,
                    rgba(139, 92, 246, 0.15),
                    transparent 30%),
                #070b14;

            color: #f8fafc;

        }


        /* ============================================================
   MAIN
============================================================ */

        .app {

            width: 100%;

            max-width: 1250px;

            margin: auto;

            padding: 35px 25px 50px;

        }


        /* ============================================================
   NAVBAR
============================================================ */

        .navbar {

            display: flex;

            justify-content: space-between;

            align-items: center;

            margin-bottom: 45px;

        }


        .logo {

            display: flex;

            align-items: center;

            gap: 12px;

        }


        .logo-icon {

            width: 46px;

            height: 46px;

            border-radius: 14px;

            display: flex;

            align-items: center;

            justify-content: center;

            font-size: 24px;

            background:
                linear-gradient(135deg,
                    #2563eb,
                    #7c3aed);

            box-shadow:
                0 10px 30px rgba(37, 99, 235, 0.35);

        }


        .logo-text {

            font-size: 20px;

            font-weight: 700;

        }


        .logo-sub {

            font-size: 12px;

            color: #94a3b8;

            margin-top: 2px;

        }


        .status {

            display: flex;

            align-items: center;

            gap: 8px;

            padding: 9px 15px;

            border-radius: 50px;

            background: rgba(34, 197, 94, 0.10);

            border: 1px solid rgba(34, 197, 94, 0.25);

            color: #86efac;

            font-size: 13px;

        }


        .status-dot {

            width: 8px;

            height: 8px;

            border-radius: 50%;

            background: #22c55e;

            box-shadow:
                0 0 10px rgba(34, 197, 94, 0.8);

        }


        /* ============================================================
   HERO
============================================================ */

        .hero {

            text-align: center;

            margin-bottom: 45px;

        }


        .badge {

            display: inline-flex;

            align-items: center;

            gap: 8px;

            padding: 8px 14px;

            border-radius: 50px;

            background:
                rgba(59, 130, 246, 0.10);

            border: 1px solid rgba(59, 130, 246, 0.25);

            color: #93c5fd;

            font-size: 13px;

            margin-bottom: 18px;

        }


        .hero h1 {

            font-size:
                clamp(38px, 6vw, 68px);

            line-height: 1.05;

            letter-spacing: -2px;

            margin-bottom: 18px;

        }


        .gradient-text {

            background:
                linear-gradient(90deg,
                    #60a5fa,
                    #a78bfa,
                    #f472b6);

            -webkit-background-clip: text;

            background-clip: text;

            color: transparent;

        }


        .hero p {

            max-width: 650px;

            margin: auto;

            color: #94a3b8;

            line-height: 1.7;

            font-size: 16px;

        }


        /* ============================================================
   DASHBOARD
============================================================ */

        .dashboard {

            display: grid;

            grid-template-columns:
                1.15fr 0.85fr;

            gap: 22px;

        }


        .card {

            background:
                rgba(15, 23, 42, 0.72);

            border: 1px solid rgba(148, 163, 184, 0.12);

            border-radius: 24px;

            backdrop-filter:
                blur(20px);

            box-shadow:
                0 25px 70px rgba(0, 0, 0, 0.30);

        }


        /* ============================================================
   UPLOAD CARD
============================================================ */

        .upload-card {

            padding: 25px;

        }


        .card-header {

            display: flex;

            justify-content: space-between;

            align-items: center;

            margin-bottom: 20px;

        }


        .card-title {

            font-size: 17px;

            font-weight: 650;

        }


        .card-label {

            font-size: 12px;

            color: #64748b;

        }


        /* ============================================================
   DROP AREA
============================================================ */

        .drop-zone {

            min-height: 390px;

            border-radius: 20px;

            border:
                1.5px dashed rgba(148, 163, 184, 0.30);

            display: flex;

            flex-direction: column;

            justify-content: center;

            align-items: center;

            text-align: center;

            padding: 30px;

            cursor: pointer;

            transition:
                0.25s ease;

            background:
                rgba(2, 6, 23, 0.35);

        }


        .drop-zone:hover {

            border-color: #60a5fa;

            background:
                rgba(59, 130, 246, 0.05);

            transform:
                translateY(-2px);

        }


        .drop-zone.dragging {

            border-color: #818cf8;

            background:
                rgba(99, 102, 241, 0.10);

            transform:
                scale(1.01);

        }


        .upload-icon {

            width: 82px;

            height: 82px;

            border-radius: 24px;

            display: flex;

            align-items: center;

            justify-content: center;

            font-size: 38px;

            margin-bottom: 22px;

            background:
                linear-gradient(135deg,
                    rgba(37, 99, 235, 0.20),
                    rgba(124, 58, 237, 0.20));

            border:
                1px solid rgba(96, 165, 250, 0.20);

        }


        .drop-zone h2 {

            font-size: 20px;

            margin-bottom: 9px;

        }


        .drop-zone p {

            color: #64748b;

            font-size: 14px;

            margin-bottom: 22px;

        }


        .supported {

            color: #475569;

            font-size: 11px;

            margin-top: 15px;

        }


        /* ============================================================
   BUTTON
============================================================ */

        .primary-btn {

            border: none;

            padding: 13px 22px;

            border-radius: 11px;

            font-size: 14px;

            font-weight: 600;

            color: white;

            cursor: pointer;

            background:
                linear-gradient(135deg,
                    #2563eb,
                    #7c3aed);

            box-shadow:
                0 8px 25px rgba(37, 99, 235, 0.25);

            transition:
                0.2s ease;

        }


        .primary-btn:hover {

            transform:
                translateY(-2px);

            box-shadow:
                0 12px 30px rgba(37, 99, 235, 0.35);

        }


        .primary-btn:disabled {

            opacity: 0.5;

            cursor: not-allowed;

            transform: none;

        }


        /* ============================================================
   RESULT CARD
============================================================ */

        .result-card {

            padding: 25px;

            display: flex;

            flex-direction: column;

        }


        .result-header {

            display: flex;

            justify-content: space-between;

            align-items: center;

            margin-bottom: 20px;

        }


        .live-label {

            font-size: 11px;

            padding: 6px 10px;

            border-radius: 20px;

            background:
                rgba(148, 163, 184, 0.08);

            color: #94a3b8;

        }


        /* ============================================================
   IMAGE PREVIEW
============================================================ */

        .preview-wrapper {

            position: relative;

            width: 100%;

            height: 255px;

            border-radius: 18px;

            overflow: hidden;

            background: #020617;

            border:
                1px solid rgba(148, 163, 184, 0.10);

        }


        .preview-wrapper img {

            width: 100%;

            height: 100%;

            object-fit: contain;

        }


        .preview-placeholder {

            width: 100%;

            height: 100%;

            display: flex;

            align-items: center;

            justify-content: center;

            color: #475569;

            font-size: 13px;

        }


        /* ============================================================
   RESULT STATUS
============================================================ */

        .result-status {

            margin-top: 20px;

            padding: 18px;

            border-radius: 16px;

            background:
                rgba(2, 6, 23, 0.5);

            border:
                1px solid rgba(148, 163, 184, 0.10);

        }


        .status-heading {

            color: #64748b;

            font-size: 11px;

            text-transform: uppercase;

            letter-spacing: 1px;

        }


        .prediction {

            font-size: 30px;

            font-weight: 750;

            margin-top: 7px;

        }


        .helmet {

            color: #4ade80;

        }


        .no-helmet {

            color: #fb7185;

        }


        /* ============================================================
   CONFIDENCE
============================================================ */

        .confidence-row {

            display: flex;

            justify-content: space-between;

            align-items: center;

            margin-top: 20px;

            margin-bottom: 9px;

        }


        .confidence-label {

            font-size: 13px;

            color: #94a3b8;

        }


        .confidence-value {

            font-size: 16px;

            font-weight: 700;

        }


        .progress {

            height: 8px;

            width: 100%;

            background:
                #1e293b;

            border-radius: 20px;

            overflow: hidden;

        }


        .progress-bar {

            height: 100%;

            width: 0%;

            border-radius: 20px;

            background:
                linear-gradient(90deg,
                    #2563eb,
                    #8b5cf6);

            transition:
                width 0.7s ease;

        }


        /* ============================================================
   INFO GRID
============================================================ */

        .info-grid {

            display: grid;

            grid-template-columns:
                1fr 1fr;

            gap: 10px;

            margin-top: 15px;

        }


        .info {

            padding: 13px;

            border-radius: 13px;

            background:
                rgba(148, 163, 184, 0.05);

        }


        .info-label {

            font-size: 10px;

            color: #64748b;

            margin-bottom: 5px;

        }


        .info-value {

            font-size: 13px;

            font-weight: 600;

        }


        /* ============================================================
   RESET BUTTON
============================================================ */

        .reset-btn {

            width: 100%;

            margin-top: 15px;

            padding: 12px;

            border-radius: 11px;

            border:
                1px solid rgba(148, 163, 184, 0.15);

            background:
                rgba(148, 163, 184, 0.05);

            color: #cbd5e1;

            cursor: pointer;

            transition: 0.2s;

        }


        .reset-btn:hover {

            background:
                rgba(148, 163, 184, 0.10);

        }


        /* ============================================================
   LOADING
============================================================ */

        .loading {

            display: none;

            align-items: center;

            gap: 10px;

            margin-top: 15px;

            color: #94a3b8;

            font-size: 13px;

        }


        .spinner {

            width: 18px;

            height: 18px;

            border-radius: 50%;

            border:
                2px solid #334155;

            border-top-color:
                #60a5fa;

            animation:
                spin 0.8s linear infinite;

        }


        @keyframes spin {

            to {

                transform:
                    rotate(360deg);

            }

        }


        /* ============================================================
   FOOTER
============================================================ */

        .footer {

            text-align: center;

            margin-top: 35px;

            color: #475569;

            font-size: 12px;

        }


        .footer span {

            color: #64748b;

        }


        /* ============================================================
   RESPONSIVE
============================================================ */

        @media(max-width: 850px) {

            .dashboard {

                grid-template-columns: 1fr;

            }

        }


        @media(max-width: 550px) {

            .app {

                padding:
                    20px 15px 35px;

            }


            .navbar {

                margin-bottom: 30px;

            }


            .status {

                display: none;

            }


            .hero h1 {

                letter-spacing: -1px;

            }


            .drop-zone {

                min-height: 320px;

            }

        }
    </style>

</head>


<body>


    <div class="app">


        <!-- ============================================================
     NAVBAR
============================================================ -->

        <nav class="navbar">


            <div class="logo">


                <div class="logo-icon">
                    🪖
                </div>


                <div>

                    <div class="logo-text">
                        Helmet AI
                    </div>

                    <div class="logo-sub">
                        Traffic Violation Detection
                    </div>

                </div>


            </div>


            <div class="status">

                <span class="status-dot"></span>

                AI Model Online

            </div>


        </nav>



        <!-- ============================================================
     HERO
============================================================ -->

        <section class="hero">


            <div class="badge">

                ✦ YOLO11 Classification Model

            </div>


            <h1>

                Intelligent

                <span class="gradient-text">
                    Helmet Detection
                </span>

            </h1>


            <p>

                Upload an image and let the trained
                AI model determine whether a helmet
                is present, along with its confidence
                score.

            </p>


        </section>



        <!-- ============================================================
     DASHBOARD
============================================================ -->

        <div class="dashboard">



            <!-- ============================================================
     UPLOAD
============================================================ -->

            <div class="card upload-card">


                <div class="card-header">

                    <div class="card-title">
                        Upload Image
                    </div>

                    <div class="card-label">
                        INPUT
                    </div>

                </div>


                <div id="dropZone" class="drop-zone">


                    <div class="upload-icon">
                        ↑
                    </div>


                    <h2>
                        Drag & Drop your image
                    </h2>


                    <p>
                        Drop an image here or browse
                        from your computer
                    </p>


                    <input type="file" id="fileInput" accept="image/*" hidden>


                    <button id="browseButton" class="primary-btn">

                        Choose Image

                    </button>


                    <div class="supported">

                        JPG · JPEG · PNG · WEBP

                    </div>


                </div>


            </div>



            <!-- ============================================================
     RESULT
============================================================ -->

            <div class="card result-card">


                <div class="result-header">

                    <div class="card-title">
                        AI Analysis
                    </div>


                    <div class="live-label">
                        READY
                    </div>

                </div>



                <!-- IMAGE -->

                <div class="preview-wrapper">


                    <div id="placeholder" class="preview-placeholder">

                        Your image preview will appear here

                    </div>


                    <img id="preview" style="display:none;">

                </div>



                <!-- LOADING -->

                <div id="loading" class="loading">

                    <div class="spinner"></div>

                    Analyzing image with AI...

                </div>



                <!-- RESULT -->

                <div id="resultStatus" class="result-status" style="display:none;">


                    <div class="status-heading">
                        Detection
                    </div>


                    <div id="prediction" class="prediction">

                    </div>



                    <div class="confidence-row">

                        <span class="confidence-label">
                            Confidence
                        </span>


                        <span id="confidence" class="confidence-value">

                            0%

                        </span>

                    </div>


                    <div class="progress">

                        <div id="progressBar" class="progress-bar">

                        </div>

                    </div>



                    <div class="info-grid">


                        <div class="info">

                            <div class="info-label">
                                MODEL
                            </div>

                            <div class="info-value">
                                YOLO11
                            </div>

                        </div>


                        <div class="info">

                            <div class="info-label">
                                CLASS
                            </div>

                            <div id="className" class="info-value">

                                --

                            </div>

                        </div>


                        <div class="info">

                            <div class="info-label">
                                PROCESSING
                            </div>

                            <div id="processingTime" class="info-value">

                                --

                            </div>

                        </div>


                        <div class="info">

                            <div class="info-label">
                                STATUS
                            </div>

                            <div id="resultState" class="info-value">

                                --

                            </div>

                        </div>


                    </div>


                </div>



                <button id="predictButton" class="primary-btn" style="display:none; width:100%; margin-top:15px;">

                    Analyze Image

                </button>


                <button id="resetButton" class="reset-btn" style="display:none;">

                    ↻ Analyze Another Image

                </button>


            </div>


        </div>



        <!-- ============================================================
     FOOTER
============================================================ -->

        <div class="footer">

            Helmet AI Detection System

            <span>
                · Powered by YOLO11
            </span>

        </div>


    </div>



    <script>


        // ============================================================
        // ELEMENTS
        // ============================================================

        const dropZone =
            document.getElementById(
                "dropZone"
            );


        const fileInput =
            document.getElementById(
                "fileInput"
            );


        const browseButton =
            document.getElementById(
                "browseButton"
            );


        const preview =
            document.getElementById(
                "preview"
            );


        const placeholder =
            document.getElementById(
                "placeholder"
            );


        const predictButton =
            document.getElementById(
                "predictButton"
            );


        const resetButton =
            document.getElementById(
                "resetButton"
            );


        const loading =
            document.getElementById(
                "loading"
            );


        const resultStatus =
            document.getElementById(
                "resultStatus"
            );


        const prediction =
            document.getElementById(
                "prediction"
            );


        const confidence =
            document.getElementById(
                "confidence"
            );


        const progressBar =
            document.getElementById(
                "progressBar"
            );


        const className =
            document.getElementById(
                "className"
            );


        const processingTime =
            document.getElementById(
                "processingTime"
            );


        const resultState =
            document.getElementById(
                "resultState"
            );


        let selectedFile = null;



        // ============================================================
        // BROWSE
        // ============================================================

        browseButton.addEventListener(
            "click",
            function (event) {

                event.stopPropagation();

                fileInput.click();

            }
        );



        // ============================================================
        // FILE INPUT
        // ============================================================

        fileInput.addEventListener(
            "change",
            function () {

                if (
                    fileInput.files.length > 0
                ) {

                    handleFile(
                        fileInput.files[0]
                    );

                }

            }
        );



        // ============================================================
        // DRAG OVER
        // ============================================================

        dropZone.addEventListener(
            "dragover",
            function (event) {

                event.preventDefault();

                dropZone.classList.add(
                    "dragging"
                );

            }
        );



        // ============================================================
        // DRAG LEAVE
        // ============================================================

        dropZone.addEventListener(
            "dragleave",
            function () {

                dropZone.classList.remove(
                    "dragging"
                );

            }
        );



        // ============================================================
        // DROP
        // ============================================================

        dropZone.addEventListener(
            "drop",
            function (event) {

                event.preventDefault();

                dropZone.classList.remove(
                    "dragging"
                );


                const files =
                    event.dataTransfer.files;


                if (
                    files.length > 0
                ) {

                    handleFile(
                        files[0]
                    );

                }

            }
        );



        // ============================================================
        // HANDLE FILE
        // ============================================================

        function handleFile(file) {


            if (
                !file.type.startsWith(
                    "image/"
                )
            ) {

                alert(
                    "Please select an image file."
                );

                return;

            }


            selectedFile = file;


            const reader =
                new FileReader();


            reader.onload =
                function (event) {


                    preview.src =
                        event.target.result;


                    preview.style.display =
                        "block";


                    placeholder.style.display =
                        "none";


                    predictButton.style.display =
                        "block";


                    resetButton.style.display =
                        "none";


                    resultStatus.style.display =
                        "none";

                };


            reader.readAsDataURL(file);

        }



        // ============================================================
        // PREDICT
        // ============================================================

        predictButton.addEventListener(
            "click",
            async function () {


                if (!selectedFile) {

                    alert(
                        "Please select an image first."
                    );

                    return;

                }


                const formData =
                    new FormData();


                formData.append(
                    "image",
                    selectedFile
                );


                // Start timer

                const start =
                    performance.now();


                // Loading

                loading.style.display =
                    "flex";


                predictButton.disabled =
                    true;


                resultStatus.style.display =
                    "none";


                try {


                    const response =
                        await fetch(
                            "/predict",
                            {
                                method: "POST",
                                body: formData
                            }
                        );


                    const data =
                        await response.json();


                    if (data.error) {

                        throw new Error(
                            data.error
                        );

                    }


                    // Processing time

                    const end =
                        performance.now();


                    const time =
                        (
                            end - start
                        ).toFixed(0);


                    // Prediction

                    prediction.textContent =
                        data.prediction;


                    // Confidence

                    const conf =
                        Number(
                            data.confidence
                        );


                    confidence.textContent =
                        conf.toFixed(2) + "%";


                    progressBar.style.width =
                        conf + "%";


                    // Class

                    className.textContent =
                        data.class;


                    // Processing

                    processingTime.textContent =
                        time + " ms";


                    // Status

                    resultState.textContent =
                        "Completed";


                    // -----------------------------------
                    // Color
                    // -----------------------------------

                    if (
                        data.prediction ===
                        "No Helmet"
                    ) {

                        prediction.className =
                            "prediction no-helmet";

                        resultState.style.color =
                            "#fb7185";

                    }

                    else {

                        prediction.className =
                            "prediction helmet";

                        resultState.style.color =
                            "#4ade80";

                    }


                    // Show result

                    resultStatus.style.display =
                        "block";


                    resetButton.style.display =
                        "block";


                    predictButton.style.display =
                        "none";


                }


                catch (error) {


                    console.error(
                        error
                    );


                    alert(
                        "Prediction failed: "
                        + error.message
                    );

                }


                finally {


                    loading.style.display =
                        "none";


                    predictButton.disabled =
                        false;

                }

            }
        );



        // ============================================================
        // RESET
        // ============================================================

        resetButton.addEventListener(
            "click",
            function () {


                selectedFile = null;

                fileInput.value = "";


                preview.src = "";

                preview.style.display =
                    "none";


                placeholder.style.display =
                    "flex";


                resultStatus.style.display =
                    "none";


                predictButton.style.display =
                    "none";


                resetButton.style.display =
                    "none";


                progressBar.style.width =
                    "0%";


                confidence.textContent =
                    "0%";

            }
        );


    </script>


</body>

</html>
"""


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return HTML_PAGE


# ============================================================
# PREDICTION API
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():


    # -----------------------------------
    # Check uploaded image
    # -----------------------------------

    if "image" not in request.files:

        return jsonify({
            "error":
            "No image uploaded."
        }), 400


    file = request.files["image"]


    if file.filename == "":

        return jsonify({
            "error":
            "No image selected."
        }), 400


    try:


        # -----------------------------------
        # Open image
        # -----------------------------------

        image = Image.open(
            file
        ).convert("RGB")


        # -----------------------------------
        # Run YOLO prediction
        # -----------------------------------

        results = model.predict(

            source=image,

            device="cpu",

            imgsz=640,

            verbose=False

        )


        result = results[0]


        # -----------------------------------
        # Get predicted class
        # -----------------------------------

        class_id = int(
            result.probs.top1
        )


        confidence = float(
            result.probs.top1conf
        )


        # IMPORTANT:
        # Keep this on ONE line

        predicted_class = result.names[class_id]


        # -----------------------------------
        # Normalize class name
        # -----------------------------------

        normalized_class = (
            predicted_class
            .lower()
            .replace(" ", "_")
        )


        # -----------------------------------
        # Final prediction
        # -----------------------------------

        if normalized_class == "no_helmet":

            prediction = "No Helmet"

        else:

            prediction = "Helmet"


        # -----------------------------------
        # Send result to browser
        # -----------------------------------

        return jsonify({

            "prediction":
                prediction,

            "class":
                predicted_class,

            "confidence":
                round(
                    confidence * 100,
                    2
                )

        })


    except Exception as e:

        print(
            "Prediction error:",
            e
        )


        return jsonify({

            "error":
                str(e)

        }), 500


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":


    print("\n")
    print("=" * 60)
    print("SERVER STARTING")
    print("=" * 60)

    print(
        "Open this in your browser:"
    )

    print(
        "http://127.0.0.1:5000"
    )

    print("=" * 60)


    app.run(

        host="0.0.0.0",

        port=5000,

        debug=False

    )