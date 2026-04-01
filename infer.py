from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
import json
import cv2
import os
from gtts import gTTS
import base64
import io

# -----------------------------
# Initialize App
# -----------------------------
app = FastAPI(title="Crop Disease Detection API")

# -----------------------------
# Enable CORS
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Base Path Fix (IMPORTANT)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "saved_models/crop_model.h5")
LABEL_MAP_PATH = os.path.join(BASE_DIR, "saved_models/label_map.npy")
DATA_JSON_PATH = os.path.join(BASE_DIR, "data1.json")

# -----------------------------
# Load Model & Data
# -----------------------------
model = tf.keras.models.load_model(MODEL_PATH)
label_map = np.load(LABEL_MAP_PATH, allow_pickle=True).item()

# Reverse mapping (index → class name)
index_to_label = {v: k for k, v in label_map.items()}

with open(DATA_JSON_PATH, "r", encoding="utf-8") as f:
    DATA = json.load(f)

print("✅ Model & Dataset Loaded Successfully")

# -----------------------------
# Helper Functions
# -----------------------------
def generate_speech(text, lang="en"):
    tts = gTTS(text=text, lang="hi" if lang == "hi" else "en")
    audio_io = io.BytesIO()
    tts.write_to_fp(audio_io)
    audio_io.seek(0)
    return base64.b64encode(audio_io.read()).decode()

def fallback_info(disease):
    return {
        "description": f"{disease} is a crop disease.",
        "cause": "Unknown cause",
        "cure": "Consult expert",
        "usage": "Follow guidelines",
        "prevention": "Maintain hygiene",
        "suggestion": "Visit agriculture expert"
    }

# -----------------------------
# MAIN API (IMAGE → DISEASE → INFO)
# -----------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...), lang: str = "en"):
    try:
        # -----------------------------
        # Read Image
        # -----------------------------
        image_bytes = await file.read()
        np_img = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if img is None:
            return JSONResponse({"error": "Invalid image file"})

        # -----------------------------
        # Preprocess Image
        # -----------------------------
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        # -----------------------------
        # Predict Disease
        # -----------------------------
        prediction = model.predict(img)[0]
        class_index = np.argmax(prediction)
        confidence = float(np.max(prediction))

        disease = index_to_label[class_index]

        # -----------------------------
        # Fetch Disease Info from JSON
        # -----------------------------
        info = DATA.get(disease, fallback_info(disease))

        # -----------------------------
        # Generate Audio
        # -----------------------------
        audio = generate_speech(info["description"], lang)

        # -----------------------------
        # Final Response
        # -----------------------------
        return {
            "disease": disease,
            "confidence": round(confidence * 100, 2),
            "description": info.get("description", ""),
            "cause": info.get("cause", ""),
            "cure": info.get("cure", ""),
            "usage": info.get("usage", ""),
            "prevention": info.get("prevention", ""),
            "suggestion": info.get("suggestion", ""),
            "audio": audio
        }

    except Exception as e:
        return JSONResponse({"error": str(e)})

# -----------------------------
# RUN SERVER
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
