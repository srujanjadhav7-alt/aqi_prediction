# src/predict.py

import tensorflow as tf
import numpy as np
import cv2
from pathlib import Path
import numpy as np
import cv2

def is_invalid_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # very low variation → blank / sky / plain image
    if np.std(gray) < 10:
        return True

    # too bright (clear sky / grass)
    if np.mean(gray) > 200:
        return True

    return False

# ── Reproducibility ───────────────────────────────────────────────────────────
tf.random.set_seed(42)
np.random.seed(42)

# ── Constants ─────────────────────────────────────────────────────────────────
IMG_SIZE = 224
AQI_MIN  = 0.0
AQI_MAX  = 500.0

# ── Load Model ────────────────────────────────────────────────────────────────
MODEL_PATH = "saved_models/aqi_model_stage1.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# ── Utility Functions ─────────────────────────────────────────────────────────
def denormalize_aqi(val: float) -> float:
    return val * (AQI_MAX - AQI_MIN) + AQI_MIN


def get_aqi_category(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"


def is_valid_image(img):
    """
    Reject blank or low-information images
    (fixed for normalized input)
    """
    # Convert back to [0,255] scale
    img_check = ((img + 1.0) * 127.5).astype("float32")

    std = np.std(img_check)

    return std > 10  # threshold tuned


def preprocess_image(image_path):
    """
    Same preprocessing as training
    """
    img = cv2.imread(str(image_path))

    if img is None:
        raise ValueError("Invalid image path")

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32")

    # Normalize to [-1, 1] (EfficientNet style)
    img = (img / 127.5) - 1.0

    return img


# ── Main Prediction Function ──────────────────────────────────────────────────
def predict_aqi(image_path):

    img = preprocess_image(image_path)

    # 🚨 Step 1: Validate image
    if not is_valid_image(img):
        return {
            "status": "error",
            "message": "Invalid image (blank or low information). Upload outdoor image."
        }

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    # 🚨 Step 2: Deterministic inference
    pred = model(img, training=False).numpy()[0][0]

    # 🚨 Step 3: Convert to AQI
    aqi = denormalize_aqi(pred)

    # 🚨 Step 4: Sanity check
    if aqi < 0 or aqi > 500:
        return {
            "status": "error",
            "message": "Unreliable prediction. Try a clearer image."
        }

    # 🚨 Step 5: Category mapping
    category = get_aqi_category(aqi)

    return {
        "status": "success",
        "aqi": round(float(aqi), 2),
        "category": category
    }


# ── Test Run ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # 🔁 Change this to test different images
    test_image = "data/raw/dawn/images/sand_storm-252.jpg"

    result = predict_aqi(test_image)
    print(result)