import os
import io
import base64
import shutil
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image

app = Flask(__name__)
CORS(app)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR     = os.path.join(BASE_DIR, "models")
MODEL_PATH    = os.path.join(MODEL_DIR, "skin_model.h5")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploaded_images")
IMG_SIZE      = (224, 224)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── Hugging Face config (set these in Render Environment tab) ─────────────────
HF_REPO_ID  = os.environ.get("HF_REPO_ID", "")
HF_FILENAME = "skin_model.h5"
HF_TOKEN    = os.environ.get("HF_TOKEN", None)

# ── Class names ───────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "Atopic Dermatitis",
    "Basal Cell",
    "Benign Keratosis",
    "Eczema",
    "Melanocytic",
    "Melanoma",
    "Psoriasis",
    "Seborrheic",
    "Tinea Ringworms Candidiasis",
    "Warts Molluscum",
]

# ── Disease info ──────────────────────────────────────────────────────────────
DISEASE_INFO = {
    "Atopic Dermatitis": {
        "emoji": "🟡", "severity": "Mild-Severe (Chronic)",
        "causes": ["Genetic mutations affecting skin barrier", "Immune system overreaction", "Environmental triggers"],
        "symptoms": ["Dry cracked skin", "Itchiness (pruritus)", "Red to brownish-gray patches"],
        "suggestions": ["Moisturize frequently", "Avoid harsh soaps", "Use prescribed topical corticosteroids"],
    },
    "Basal Cell": {
        "emoji": "🔴", "severity": "Moderate (Skin Cancer)",
        "causes": ["Long-term UV radiation exposure", "Fair skin type", "History of sunburns"],
        "symptoms": ["Pearly or waxy bump", "Flat flesh-colored lesion", "Bleeding sore that heals and returns"],
        "suggestions": ["See a dermatologist for biopsy and removal", "Avoid direct sun", "Use sunscreen daily"],
    },
    "Benign Keratosis": {
        "emoji": "🟤", "severity": "Mild (Non-cancerous)",
        "causes": ["Genetic predisposition", "Aging", "Sun exposure"],
        "symptoms": ["Waxy scaly appearance", "Pasted-on look", "Light tan to black color"],
        "suggestions": ["Usually no treatment needed", "Consult doctor if irritated", "Do not scratch"],
    },
    "Eczema": {
        "emoji": "🟡", "severity": "Mild-Severe (Chronic)",
        "causes": ["Genetic predisposition", "Immune system overreaction", "Allergens: dust mites, pollen"],
        "symptoms": ["Intense itching", "Dry cracked patches", "Red-to-gray patches"],
        "suggestions": ["Moisturize twice daily", "Use mild unscented soaps", "Avoid triggers"],
    },
    "Melanocytic": {
        "emoji": "⚫", "severity": "Mild-Moderate (Moles)",
        "causes": ["Overgrowth of melanocytes", "Genetics", "Sun exposure"],
        "symptoms": ["Uniform color", "Round or oval shape", "Flat or raised"],
        "suggestions": ["Monitor using ABCDE rule", "Apply sunscreen", "See dermatologist if changes occur"],
    },
    "Melanoma": {
        "emoji": "🔴", "severity": "Potentially Life-Threatening",
        "causes": ["Excessive UV radiation", "Genetic mutations", "Family history of melanoma"],
        "symptoms": ["Asymmetric mole", "Irregular border", "Multiple colors in lesion"],
        "suggestions": ["See a dermatologist IMMEDIATELY", "Full-body skin examination", "Apply SPF 30+ daily"],
    },
    "Psoriasis": {
        "emoji": "🟡", "severity": "Moderate-Severe (Chronic)",
        "causes": ["Autoimmune condition", "Genetic predisposition", "Stress or infections"],
        "symptoms": ["Thick red patches with silver scales", "Dry cracked skin", "Thickened nails"],
        "suggestions": ["Keep skin moisturized", "Use topical corticosteroids", "Avoid known triggers"],
    },
    "Seborrheic": {
        "emoji": "🟠", "severity": "Mild",
        "causes": ["Malassezia yeast overgrowth", "Excess oil production", "Stress and fatigue"],
        "symptoms": ["Flaky scales (dandruff)", "Greasy patches", "Mild itching"],
        "suggestions": ["Use medicated shampoos", "Wash with ketoconazole shampoo", "Manage stress"],
    },
    "Tinea Ringworms Candidiasis": {
        "emoji": "🟢", "severity": "Mild (Contagious)",
        "causes": ["Fungal infection", "Skin-to-skin contact", "Warm moist environments"],
        "symptoms": ["Ring-shaped rash", "Itchy red scaly skin", "Blisters"],
        "suggestions": ["Keep area clean and dry", "Use antifungal creams", "Avoid sharing personal items"],
    },
    "Warts Molluscum": {
        "emoji": "🦠", "severity": "Mild (Contagious)",
        "causes": ["HPV virus", "Molluscum contagiosum virus", "Direct contact"],
        "symptoms": ["Small rough bumps", "Raised bumps with dimple", "Can appear anywhere"],
        "suggestions": ["Avoid scratching", "Use OTC wart treatments", "Consult doctor for removal"],
    },
}

# ── Model loading ─────────────────────────────────────────────────────────────
model        = None
load_error   = ""
model_status = "loading"


def download_from_hf():
    global load_error
    if not HF_REPO_ID:
        load_error = "HF_REPO_ID environment variable is not set in Render"
        print(f"❌ {load_error}")
        return False
    try:
        from huggingface_hub import hf_hub_download
        print(f"⬇️  Downloading model from Hugging Face: {HF_REPO_ID}/{HF_FILENAME}")
        downloaded = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_FILENAME,
            token=HF_TOKEN,
        )
        shutil.copy(downloaded, MODEL_PATH)
        print(f"✅ Model downloaded to: {MODEL_PATH}")
        return True
    except Exception as e:
        load_error = str(e)
        print(f"❌ Hugging Face download failed: {e}")
        return False

def load_skin_model():
    global model, load_error, model_status
    model_status = "loading"


    if not os.path.exists(MODEL_PATH):
        ok = download_from_hf()
        if not ok:
            return

    try:
    import tensorflow as tf

    print(f"📂 Loading model from: {MODEL_PATH}")

    model = tf.keras.models.load_model(MODEL_PATH)

    print(f"✅ Model loaded — output classes: {model.output_shape[-1]}")

    load_error = ""
    model_status = "ready"

except Exception as e:
    load_error = str(e)
    print(f"❌ Could not load model: {e}")

    # ❌ DO NOT DELETE MODEL FILE
    # if os.path.exists(MODEL_PATH):
    #     os.remove(MODEL_PATH)

    model = None   # ✅ FIXED
    model_status = "failed"

# ── Start model loading in background (port binds immediately) ────────────────
threading.Thread(target=load_skin_model, daemon=True).start()


# ── Helpers ───────────────────────────────────────────────────────────────────
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def get_info(prediction):
    if prediction in DISEASE_INFO:
        return DISEASE_INFO[prediction]
    for key, val in DISEASE_INFO.items():
        if key.lower() in prediction.lower():
            return val
    return {
        "emoji": "🔬", "severity": "Unknown",
        "causes": ["Information not available"],
        "symptoms": ["Consult a dermatologist"],
        "suggestions": ["See a certified dermatologist"],
    }


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


# ── DEBUG endpoint — visit /debug in browser to see model status ──────────────
@app.route("/debug")
def debug():
    return jsonify({
        "model_loaded":   model is not None,
        "model_path":     MODEL_PATH,
        "model_exists_on_disk": os.path.exists(MODEL_PATH),
        "HF_REPO_ID":     HF_REPO_ID or "NOT SET",
        "HF_TOKEN_set":   HF_TOKEN is not None,
        "load_error":     load_error or "none",
    })


@app.route("/predict", methods=["POST"])
def predict():
    if model_status == "loading":
        return jsonify({
            "error": "Model is still loading, please wait 30 seconds and try again."
        }), 503

    if model is None:
        return jsonify({
            "error": f"Model failed to load. Reason: {load_error}. Visit /debug for details."
        }), 500

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    allowed = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        return jsonify({"error": f"Unsupported file type: {ext}"}), 400

    image_bytes = file.read()
    save_path   = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(save_path, "wb") as f:
        f.write(image_bytes)

    media_type     = "image/jpeg" if ext in {".jpg", ".jpeg"} else f"image/{ext.lstrip('.')}"
    image_data_url = f"data:{media_type};base64,{base64.b64encode(image_bytes).decode()}"

    try:
        preds = model.predict(preprocess_image(image_bytes))[0]
        n     = len(preds)
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
            "info":       get_info(prediction),
        })

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)