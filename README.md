# 🤖 AI Image Detector

A production-ready full-stack web application that detects whether an image was **AI-generated** or **real**, powered by a HuggingFace pretrained model served via Flask.

---

## ✨ Features

- **Drag-and-drop or click-to-upload** image interface
- Supports **JPG, PNG, WEBP** up to **5 MB**
- Real-time **progress bar** and **loading animation** with step indicators
- Returns: **prediction label**, **confidence %**, and a **human-readable explanation**
- **Dark glassmorphism UI** — fully responsive, mobile-first
- Model loaded **once at startup** (not per request) for fast inference
- `/health` endpoint for deployment platform health checks
- CORS enabled, ready for Render / Railway / Fly.io

---

## 📁 Project Structure

```
ai-detector/
├── app.py               # Flask backend + model loading + /predict route
├── requirements.txt     # Python dependencies
├── Procfile             # Gunicorn entrypoint for deployment
├── .env.example         # Environment variable template
├── templates/
│   └── index.html       # Complete frontend (HTML + CSS + JS in one file)
└── static/              # (optional) additional static assets
```

---

## 🚀 Quick Start (Local)

### 1. Clone & enter directory
```bash
git clone <your-repo>
cd ai-detector
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> ⚠️ First install will download the `umm-maybe/AI-image-detector` model (~200MB) from HuggingFace.

### 4. Configure environment
```bash
cp .env.example .env
# Edit .env if needed
```

### 5. Run the server
```bash
python app.py
```

Open **http://localhost:5000** in your browser.

---

## 🌐 Deploy to Render

1. Push project to a GitHub repo
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your repo
4. Set:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120`
5. Add environment variable: `MODEL_NAME=umm-maybe/AI-image-detector`
6. Choose at least a **Standard** instance (model needs ~1GB RAM)

---

## 🌐 Deploy to Railway

1. Install Railway CLI: `npm i -g @railway/cli`
2. `railway login && railway init`
3. `railway up`
4. Set env var: `MODEL_NAME=umm-maybe/AI-image-detector`

---

## 📡 API Reference

### `POST /predict`

Upload an image for AI detection.

**Request** — `multipart/form-data`:
| Field | Type | Description |
|-------|------|-------------|
| `image` | File | JPG, PNG, or WEBP image (max 5 MB) |

**Response** — `application/json`:
```json
{
  "prediction": "AI Generated",
  "confidence": 97.43,
  "explanation": "The model is highly confident (97.4%) this image was created by an AI system..."
}
```

| Field | Type | Values |
|-------|------|--------|
| `prediction` | string | `"AI Generated"` or `"Real"` |
| `confidence` | float | `0.0` – `100.0` (percentage) |
| `explanation` | string | Human-readable analysis |

**Error response**:
```json
{ "error": "Descriptive error message" }
```

### `GET /health`
```json
{ "status": "ok", "model_loaded": true, "device": "cpu" }
```

---

## 🧠 Model

Uses [`umm-maybe/AI-image-detector`](https://huggingface.co/umm-maybe/AI-image-detector) from HuggingFace — a fine-tuned ViT (Vision Transformer) model trained to distinguish AI-generated images from real photographs.

To use a different model, change the `MODEL_NAME` environment variable. The model must be compatible with `transformers.pipeline("image-classification")`.

---

## ⚙️ Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `5000` | Server port |
| `FLASK_DEBUG` | `false` | Enable debug mode |
| `MODEL_NAME` | `umm-maybe/AI-image-detector` | HuggingFace model ID |
| `HUGGINGFACE_TOKEN` | — | (Optional) For private models |

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| Flask | Web framework |
| flask-cors | Cross-Origin Resource Sharing |
| Pillow | Image loading and preprocessing |
| torch | PyTorch inference backend |
| transformers | HuggingFace model pipeline |
| gunicorn | Production WSGI server |

---

## 📄 License

MIT — free to use and modify.
