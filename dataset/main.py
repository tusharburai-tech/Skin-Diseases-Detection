"""
main.py — Single entry point for Gunicorn and local dev.

Gunicorn (render.yaml):  gunicorn main:app --config gunicorn.conf.py
Local dev:               python main.py
"""

import os
import io
import base64
import shutil
import threading
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image

# ── Resolve project root ───────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Flask app ──────────────────────────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)
CORS(app)

# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_DIR     = os.path.join(BASE_DIR, "models")
MODEL_PATH    = os.path.join(MODEL_DIR, "skin_model.h5")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploaded_images")
IMG_SIZE      = (224, 224)

os.makedirs(MODEL_DIR,     exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── Hugging Face config (set these in Render → Environment) ───────────────────
HF_REPO_ID  = os.environ.get("HF_REPO_ID", "").strip()
HF_FILENAME = "skin_model.h5"
HF_TOKEN    = os.environ.get("HF_TOKEN", "").strip() or None

# ── Class names ────────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "Atopic Dermatitis", "Basal Cell", "Benign Keratosis", "Eczema",
    "Melanocytic", "Melanoma", "Psoriasis", "Seborrheic",
    "Tinea Ringworms Candidiasis", "Warts Molluscum",
]

# ── Disease info ───────────────────────────────────────────────────────────────
DISEASE_INFO = {
    "Atopic Dermatitis": {
        "emoji": "🟡", "severity": "Mild-Severe (Chronic)",
        "causes":      ["Genetic mutations affecting skin barrier", "Immune system overreaction", "Environmental triggers"],
        "symptoms":    ["Dry cracked skin", "Itchiness (pruritus)", "Red to brownish-gray patches"],
        "suggestions": ["Moisturize frequently", "Avoid harsh soaps", "Use prescribed topical corticosteroids"],
    },
    "Basal Cell": {
        "emoji": "🔴", "severity": "Moderate (Skin Cancer)",
        "causes":      ["Long-term UV radiation exposure", "Fair skin type", "History of sunburns"],
        "symptoms":    ["Pearly or waxy bump", "Flat flesh-colored lesion", "Bleeding sore that heals and returns"],
        "suggestions": ["⚠️ See a dermatologist for biopsy and removal", "Avoid direct sun", "Use sunscreen daily"],
    },
    "Benign Keratosis": {
        "emoji": "🟤", "severity": "Mild (Non-cancerous)",
        "causes":      ["Genetic predisposition", "Aging", "Sun exposure"],
        "symptoms":    ["Waxy scaly appearance", "Pasted-on look", "Light tan to black color"],
        "suggestions": ["Usually no treatment needed", "Consult doctor if irritated", "Do not scratch"],
    },
    "Eczema": {
        "emoji": "🟡", "severity": "Mild-Severe (Chronic)",
        "causes":      ["Genetic predisposition", "Immune system overreaction", "Allergens: dust mites, pollen"],
        "symptoms":    ["Intense itching", "Dry cracked patches", "Red-to-gray patches"],
        "suggestions": ["Moisturize twice daily", "Use mild unscented soaps", "Avoid triggers"],
    },
    "Melanocytic": {
        "emoji": "⚫", "severity": "Mild-Moderate (Moles)",
        "causes":      ["Overgrowth of melanocytes", "Genetics", "Sun exposure"],
        "symptoms":    ["Uniform color", "Round or oval shape", "Flat or raised"],
        "suggestions": ["Monitor using ABCDE rule", "Apply sunscreen", "See dermatologist if changes occur"],
    },
    "Melanoma": {
        "emoji": "🔴", "severity": "Potentially Life-Threatening",
        "causes":      ["Excessive UV radiation", "Genetic mutations", "Family history of melanoma"],
        "symptoms":    ["Asymmetric mole", "Irregular border", "Multiple colors in lesion"],
        "suggestions": ["⚠️ See a dermatologist IMMEDIATELY", "Full-body skin examination", "Apply SPF 30+ daily"],
    },
    "Psoriasis": {
        "emoji": "🟡", "severity": "Moderate-Severe (Chronic)",
        "causes":      ["Autoimmune condition", "Genetic predisposition", "Stress or infections"],
        "symptoms":    ["Thick red patches with silver scales", "Dry cracked skin", "Thickened nails"],
        "suggestions": ["Keep skin moisturized", "Use topical corticosteroids", "Avoid known triggers"],
    },
    "Seborrheic": {
        "emoji": "🟠", "severity": "Mild",
        "causes":      ["Malassezia yeast overgrowth", "Excess oil production", "Stress and fatigue"],
        "symptoms":    ["Flaky scales (dandruff)", "Greasy patches", "Mild itching"],
        "suggestions": ["Use medicated shampoos", "Wash with ketoconazole shampoo", "Manage stress"],
    },
    "Tinea Ringworms Candidiasis": {
        "emoji": "🟢", "severity": "Mild (Contagious)",
        "causes":      ["Fungal infection", "Skin-to-skin contact", "Warm moist environments"],
        "symptoms":    ["Ring-shaped rash", "Itchy red scaly skin", "Blisters"],
        "suggestions": ["Keep area clean and dry", "Use antifungal creams", "Avoid sharing personal items"],
    },
    "Warts Molluscum": {
        "emoji": "🦠", "severity": "Mild (Contagious)",
        "causes":      ["HPV virus", "Molluscum contagiosum virus", "Direct contact"],
        "symptoms":    ["Small rough bumps", "Raised bumps with dimple", "Can appear anywhere"],
        "suggestions": ["Avoid scratching", "Use OTC wart treatments", "Consult doctor for removal"],
    },
}

_FALLBACK_INFO = {
    "emoji": "🔬", "severity": "Unknown",
    "causes":      ["Information not available for this condition."],
    "symptoms":    ["Please consult a dermatologist for proper diagnosis."],
    "suggestions": ["Consult a certified dermatologist immediately."],
}


def _get_info(prediction: str) -> dict:
    if prediction in DISEASE_INFO:
        return DISEASE_INFO[prediction]
    pred_lower = prediction.lower()
    for key, val in DISEASE_INFO.items():
        if key.lower() in pred_lower or pred_lower in key.lower():
            return val
    return _FALLBACK_INFO


# ── Thread-safe model state ────────────────────────────────────────────────────
_lock         = threading.Lock()
_model        = None
_load_error   = ""
_model_status = "loading"   # "loading" | "ready" | "failed"


def _set_state(status, error="", mdl=None):
    global _model, _load_error, _model_status
    with _lock:
        _model_status = status
        _load_error   = error
        if mdl is not None:
            _model = mdl


def _get_state():
    with _lock:
        return _model_status, _load_error, _model


# ── Background model loader ────────────────────────────────────────────────────
def _download_from_hf() -> bool:
    """Download model from Hugging Face. Returns True on success."""
    if not HF_REPO_ID:
        msg = "HF_REPO_ID env var is not set — add it in Render → Environment"
        _set_state("failed", msg)
        print(f"❌ {msg}")
        return False
    try:
        from huggingface_hub import hf_hub_download
        print(f"⬇️  Downloading model: {HF_REPO_ID}/{HF_FILENAME}")
        downloaded = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_FILENAME,
            token=HF_TOKEN,
        )
        shutil.copy(downloaded, MODEL_PATH)
        print(f"✅ Model saved → {MODEL_PATH}")
        return True
    except Exception as exc:
        _set_state("failed", f"HuggingFace download failed: {exc}")
        print(f"❌ Download failed: {exc}")
        return False


def _load_model_thread():
    """
    Runs in a daemon thread at startup.
    Downloads the model if missing, then loads it into memory.
    TensorFlow is imported HERE (not at module level) so Gunicorn
    binds the port instantly without waiting for TF to initialise.
    """
    # Step 1: Download if model file is missing
    if not os.path.exists(MODEL_PATH):
        ok = _download_from_hf()
        if not ok:
            return   # state already set to "failed" inside _download_from_hf

    # Step 2: Load into memory
    try:
        import tensorflow as tf   # intentionally deferred — keeps startup fast
        print("📂 Loading model into memory…")
        mdl = tf.keras.models.load_model(MODEL_PATH)
        print(f"✅ Model ready — output classes: {mdl.output_shape[-1]}")
        _set_state("ready", mdl=mdl)
    except Exception as exc:
        # Remove the file so the next deploy re-downloads a fresh copy
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        _set_state("failed", f"Keras load error: {exc}")
        print(f"❌ Model load failed: {exc}")


# Start the background thread the moment this module is imported.
# Gunicorn finishes binding the port before the thread even starts,
# so Render's health-check always passes immediately.
threading.Thread(
    target=_load_model_thread,
    daemon=True,
    name="model-loader",
).start()


# ── Image preprocessing ────────────────────────────────────────────────────────
def _preprocess(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    """
    Polled by the frontend every 5 s to know when the model is ready.
    Returns:
        { "status": "loading" }  — model still downloading / loading
        { "status": "ready"   }  — predictions available
        { "status": "failed"  }  — model could not be loaded (check /debug)
    """
    status, error, mdl = _get_state()
    return jsonify({
        "status":      status,
        "model_ready": mdl is not None,
        "error":       error or None,
    })


@app.route("/debug")
def debug():
    """
    Diagnostic endpoint — visit  https://<your-app>.onrender.com/debug
    to inspect model state without reading Render logs.
    """
    status, error, mdl = _get_state()
    return jsonify({
        "model_loaded":         mdl is not None,
        "model_status":         status,
        "model_path":           MODEL_PATH,
        "model_exists_on_disk": os.path.exists(MODEL_PATH),
        "HF_REPO_ID":           HF_REPO_ID or "NOT SET ⚠️",
        "HF_TOKEN_set":         HF_TOKEN is not None,
        "load_error":           error or "none",
    })


@app.route("/predict", methods=["POST"])
def predict():
    status, error, mdl = _get_state()

    # ── Guard: still loading ──────────────────────────────────────────────────
    if status == "loading":
        return jsonify({
            "error": "Model is still loading. Please wait 30 seconds and try again."
        }), 503   # 503 = frontend knows to retry, not crash

    # ── Guard: load failed ────────────────────────────────────────────────────
    if status == "failed" or mdl is None:
        return jsonify({
            "error": f"Model failed to load: {error}. Visit /debug for details."
        }), 500

    # ── Validate uploaded file ────────────────────────────────────────────────
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded. Please attach a file."}), 400

    file = request.files["image"]
    if not file or file.filename == "":
        return jsonify({"error": "File has an empty filename."}), 400

    allowed_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_ext:
        return jsonify({
            "error": f"Unsupported file type '{ext}'. Please upload JPG, PNG, or WEBP."
        }), 400

    # ── Run inference ─────────────────────────────────────────────────────────
    try:
        image_bytes = file.read()

        # Persist uploaded image to disk
        save_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(save_path, "wb") as fh:
            fh.write(image_bytes)

        # Build base64 data-URL so the frontend can show the image back
        media_type     = "image/jpeg" if ext in {".jpg", ".jpeg"} else f"image/{ext.lstrip('.')}"
        image_data_url = f"data:{media_type};base64,{base64.b64encode(image_bytes).decode()}"

        # Predict
        preds  = mdl.predict(_preprocess(image_bytes))[0]
        n      = len(preds)
        labels = CLASS_NAMES[:n] if len(CLASS_NAMES) >= n \
            else CLASS_NAMES + [f"Class_{i}" for i in range(len(CLASS_NAMES), n)]

        top_idx    = int(np.argmax(preds))
        confidence = round(float(preds[top_idx]) * 100, 2)
        prediction = labels[top_idx]
        top3 = [
            {"label": labels[i], "confidence": round(float(preds[i]) * 100, 2)}
            for i in np.argsort(preds)[::-1][:3]
        ]

        return jsonify({
            "prediction": prediction,
            "confidence": confidence,
            "top3":       top3,
            "image":      image_data_url,
            "info":       _get_info(prediction),
        })

    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {exc}"}), 500


# ── Local dev entry point ──────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)