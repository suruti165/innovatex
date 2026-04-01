from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
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
app = FastAPI(title="Crop Disease Prediction API")

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
# Paths
# -----------------------------
MODEL_PATH = "saved_models/crop_model.h5"
LABEL_MAP_PATH = "saved_models/label_map.npy"
DATA_JSON_PATH = "data1.json"

# -----------------------------
# Load Model & Data
# -----------------------------
model = tf.keras.models.load_model(MODEL_PATH)
label_map = np.load(LABEL_MAP_PATH, allow_pickle=True).item()

with open(DATA_JSON_PATH, "r", encoding="utf-8") as f:
    DATA = json.load(f)

print("✅ Model & Data Loaded")

# -----------------------------
# Helper Functions
# -----------------------------
def generate_speech(text, lang="en"):
    tts = gTTS(text=text, lang="hi" if lang == "hi" else "en")
    audio_io = io.BytesIO()
    tts.write_to_fp(audio_io)
    audio_io.seek(0)
    return base64.b64encode(audio_io.read()).decode()

def translate(text, lang):
    if lang == "hi":
        return f"हिंदी विवरण: {text}"
    return text

def fallback_info(disease):
    return {
        "description": f"{disease} is a common crop disease.",
        "cause": "Pathogen infection or pest attack.",
        "cure": "Use recommended fungicide or pesticide.",
        "usage": "Follow label instructions.",
        "prevention": "Crop rotation and field hygiene.",
        "suggestion": "Consult agriculture expert."
    }

# -----------------------------
# PREDICT API
# -----------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...), lang: str = "en"):
    try:
        image_bytes = await file.read()
        np_img = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if img is None:
            return JSONResponse({"error": "Invalid image"})

        img = cv2.resize(img, (224, 224)) / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img)[0]
        disease = list(label_map.keys())[np.argmax(prediction)]

        info = DATA.get(disease, fallback_info(disease))

        audio = generate_speech(info["description"], lang)

        return {
            "disease": disease,
            "description": translate(info.get("description", ""), lang),
            "cause": translate(info.get("cause", ""), lang),
            "cure": translate(info.get("cure", ""), lang),
            "usage": translate(info.get("usage", ""), lang),
            "prevention": translate(info.get("prevention", ""), lang),
            "suggestion": translate(info.get("suggestion", ""), lang),
            "audio": audio
        }

    except Exception as e:
        return JSONResponse({"error": str(e)})

# -----------------------------
# ASK API
# -----------------------------
@app.get("/ask")
def ask(question: str, lang: str = "en"):
    q = question.lower()

    for disease, info in DATA.items():
        if disease.lower() in q:
            audio = generate_speech(info["description"], lang)
            return {
                "description": translate(info["description"], lang),
                "audio": audio
            }

    audio = generate_speech("No disease information found", lang)
    return {
        "reply": "No disease information found",
        "audio": audio
    }

# -----------------------------
# RUN SERVER
# -----------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)