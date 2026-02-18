"""
AI Image Detection Web Application
Backend: Flask + HuggingFace Transformers
"""

import os
import io
import logging
from typing import Tuple
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
import torch
from transformers import pipeline
import warnings

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(
    __name__,
    static_folder=os.path.join(BASE_DIR, "static"),
    template_folder=os.path.join(BASE_DIR, "templates")
)
CORS(app)

MAX_FILE_SIZE = 5 * 1024 * 1024
ALLOWED_EXTENSIONS = {"image/jpeg", "image/png", "image/webp"}
MODEL_NAME = os.getenv("MODEL_NAME", "umm-maybe/AI-image-detector")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logger.info(f"Loading model '{MODEL_NAME}' on device '{DEVICE}'...")
try:
    classifier = pipeline(
        "image-classification",
        model=MODEL_NAME,
        device=0 if DEVICE == "cuda" else -1,
    )
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    classifier = None


def validate_image(file) -> Tuple[bool, str]:
    content_type = file.content_type
    if content_type not in ALLOWED_EXTENSIONS:
        return False, f"Invalid file type '{content_type}'. Only JPG, PNG, WEBP allowed."
    file.seek(0, os.SEEK_END)
    size = file.tell()
    file.seek(0)
    if size > MAX_FILE_SIZE:
        return False, f"File too large ({size / 1024 / 1024:.2f} MB). Max is 5 MB."
    return True, ""


def build_explanation(label: str, confidence: float) -> str:
    pct = round(confidence * 100, 1)
    if label == "AI Generated":
        if confidence >= 0.90:
            return f"The model is highly confident ({pct}%) this image was created by an AI system. Telltale signs include hyper-smooth textures, unnatural lighting gradients, and structural inconsistencies typical of generative models."
        elif confidence >= 0.70:
            return f"The model suspects ({pct}%) this image is AI-generated. Several subtle artifacts such as irregular edges or blended features are characteristic of diffusion or GAN-based synthesis."
        else:
            return f"The model leans toward AI-generated ({pct}%), but with moderate uncertainty. The image has some synthetic-looking qualities, though it shares traits with real photography too."
    else:
        if confidence >= 0.90:
            return f"The model is highly confident ({pct}%) this is a real photograph. Natural noise patterns, authentic lighting, and organic imperfections are consistent with a camera-captured image."
        elif confidence >= 0.70:
            return f"The model believes ({pct}%) this is a real image. It exhibits mostly authentic photographic characteristics, with only minor ambiguities."
        else:
            return f"The model leans toward real ({pct}%), but is not highly certain. This image sits near the boundary between AI-generated and real photography."


@app.route("/")
def index():
    try:
        return render_template("index.html")
    except Exception as e:
        logger.error(f"Template error: {e}")
        return f"<h1>Template Error</h1><p>{str(e)}</p><p>Looking in: {app.template_folder}</p>", 500


@app.route("/predict", methods=["POST"])
def predict():
    if classifier is None:
        return jsonify({"error": "Model not loaded. Please check server logs."}), 503
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400
    valid, msg = validate_image(file)
    if not valid:
        return jsonify({"error": msg}), 422
    try:
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({"error": "Could not read image. File may be corrupted."}), 422
    try:
        results = classifier(image)
        logger.info(f"Raw model output: {results}")
    except Exception as e:
        logger.error(f"Inference error: {e}")
        return jsonify({"error": "Model inference failed."}), 500

    top = max(results, key=lambda x: x["score"])
    raw_label = top["label"].lower()
    confidence = round(float(top["score"]), 4)

    if any(kw in raw_label for kw in ["artificial", "fake", "ai", "generated", "synthetic"]):
        prediction = "AI Generated"
    else:
        prediction = "Real"

    explanation = build_explanation(prediction, confidence)
    response = {
        "prediction": prediction,
        "confidence": round(confidence * 100, 2),
        "explanation": explanation,
    }
    logger.info(f"Prediction: {prediction} | Confidence: {confidence * 100:.2f}%")
    return jsonify(response), 200


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": classifier is not None,
        "device": DEVICE,
        "template_folder": app.template_folder,
        "template_exists": os.path.exists(os.path.join(app.template_folder, "index.html")),
    }), 200


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    logger.info(f"Template folder: {app.template_folder}")
    logger.info(f"index.html exists: {os.path.exists(os.path.join(app.template_folder, 'index.html'))}")
    app.run(host="0.0.0.0", port=port, debug=False)