# ============================================
# HELMET CLASSIFICATION - TESTING
# ============================================

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
from io import BytesIO

import requests
import matplotlib.pyplot as plt
from PIL import Image

from ultralytics import YOLO


# ============================================
# 1. Load Trained Model
# ============================================

model_path = Path("saved_models/helmet_model.pt")

if not model_path.exists():
    raise FileNotFoundError(
        f"Model not found at: {model_path.resolve()}"
    )

model = YOLO(str(model_path))

print("============================================")
print("HELMET CLASSIFICATION - MODEL TESTING")
print("============================================")

print("\nModel loaded successfully!")
print("Model:", model_path.resolve())
print("Classes:", model.names)


# ============================================
# 2. Image URL
# ============================================

image_url = "PASTE_IMAGE_URL_HERE"


# ============================================
# 3. Download Image
# ============================================

def download_image(url):

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=15
    )

    response.raise_for_status()

    image = Image.open(
        BytesIO(response.content)
    ).convert("RGB")

    return image


try:

    image = download_image(image_url)

    print("\nImage downloaded successfully!")
    print("Image size:", image.size)


except Exception as e:

    print("\nError downloading image:")
    print(e)

    raise


# ============================================
# 4. Run Model Prediction
# ============================================

results = model.predict(
    source=image,
    verbose=False
)

result = results[0]


# ============================================
# 5. Extract Prediction
# ============================================

top1_class = int(result.probs.top1)

confidence = float(
    result.probs.top1conf
)

predicted_class = model.names[top1_class]


# ============================================
# 6. Display Result
# ============================================

print("\n============================================")
print("              TEST RESULT")
print("============================================")

print("Predicted Class :", predicted_class)
print("Confidence      :", f"{confidence:.2%}")




# ============================================
# 7. Display Image
# ============================================

plt.figure(figsize=(10, 7))

plt.imshow(image)
plt.axis("off")

plt.title(
    f"Prediction: {predicted_class}\n"
    f"Confidence: {confidence:.2%}",
    fontsize=16
)

plt.show()