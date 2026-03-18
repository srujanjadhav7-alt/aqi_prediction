##Docs UI:    http://127.0.0.1:8000/docs

import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import io
from PIL import Image
import logging

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
IMG_SIZE    = 224
AQI_MIN     = 0.0
AQI_MAX     = 500.0
MODEL_PATH  = PROJECT_ROOT / "saved_models" / "aqi_model_final.h5"

# ── AQI Category Helper ────────────────────────────────────────────────────────
AQI_CATEGORIES = [
    (0,   50,  "Good",                          "Air quality is satisfactory."),
    (51,  100, "Moderate",                       "Acceptable air quality."),
    (101, 150, "Unhealthy for Sensitive Groups", "Sensitive groups may be affected."),
    (151, 200, "Unhealthy",                      "Everyone may experience effects."),
    (201, 300, "Very Unhealthy",                 "Health alert for everyone."),
    (301, 500, "Hazardous",                      "Emergency health conditions."),
]

def get_aqi_info(aqi: float) -> dict:
    for lo, hi, cat, desc in AQI_CATEGORIES:
        if lo <= aqi <= hi:
            return {"category": cat, "description": desc}
    return {"category": "Unknown", "description": "Out of range."}


def denormalize_aqi(val: float) -> float:
    return float(val) * (AQI_MAX - AQI_MIN) + AQI_MIN


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Loads image bytes, resizes to 224x224, applies EfficientNet preprocessing."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32)
    arr = (arr / 127.5) - 1.0           # EfficientNet normalization
    arr = np.expand_dims(arr, axis=0)   # Add batch dimension → (1, 224, 224, 3)
    return arr


# ── App Setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AQI Prediction API",
    description="Predicts Air Quality Index from outdoor images using EfficientNetB3",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load Model on Startup ──────────────────────────────────────────────────────
model = None

@app.on_event("startup")
async def load_model():
    global model
    log.info(f"Loading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(str(MODEL_PATH))
    log.info("Model loaded successfully ✅")


# ── Response Schema ────────────────────────────────────────────────────────────
class AQIPrediction(BaseModel):
    aqi:         float
    category:    str
    description: str
    confidence:  str


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "AQI Prediction API is running. POST an image to /predict"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=AQIPrediction)
async def predict(file: UploadFile = File(...)):
    """
    Upload an outdoor image (JPG/PNG) and get a predicted AQI value.
    """
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload a JPG or PNG image."
        )

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    try:
        image_bytes = await file.read()
        img_array   = preprocess_image(image_bytes)

        # Run inference
        raw_output  = model.predict(img_array, verbose=0)[0][0]

        # Clamp output to [0, 1] before denormalizing
        raw_output  = float(np.clip(raw_output, 0.0, 1.0))
        aqi_value   = round(denormalize_aqi(raw_output), 1)

        info = get_aqi_info(aqi_value)

        log.info(f"Predicted AQI: {aqi_value} ({info['category']})")

        return AQIPrediction(
            aqi=aqi_value,
            category=info["category"],
            description=info["description"],
            confidence="Model MAE ≈ 80 AQI points"
        )

    except Exception as e:
        log.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    