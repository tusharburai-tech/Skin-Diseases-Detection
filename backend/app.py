import os
import base64
import threading
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename

from backend.model_loader import load_skin_model
from backend.predict import predict_image

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder="../templates", static_folder="../static")
CORS(app)

BASE_DIR      = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "..", "static", "uploaded_images")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── Load model in background thread (port binds immediately) ─────────────────
model        = None
model_status = "loading"
load_error   = ""


def _load_model_bg():
    global model, model_status, load_error
    try:
        result = load_skin_model()
        if result is None:
            model_status = "failed"
            load_error   = "load_skin_model() returned None"
        else:
            model        = result
            model_status = "ready"
            load_error   = ""
    except Exception as e:
        model_status = "failed"
        load_error   = str(e)


threading.Thread(target=_load_model_bg, daemon=True).start()

# ── Disease information ───────────────────────────────────────────────────────
DISEASE_INFO = {
    "Atopic Dermatitis": {
        "emoji": "🟡", "severity": "Mild–Severe (Chronic)",
        "causes": ["Genetic mutations affecting skin barrier", "Immune system overreaction", "Environmental triggers"],
        "symptoms": ["Dry, cracked skin", "Itchiness (pruritus)", "Red to brownish-gray patches"],
        "suggestions": ["Moisturize frequently", "Avoid harsh soaps", "Use prescribed topical corticosteroids"],
    },
    "Basal Cell": {
        "emoji": "🔴", "severity": "Moderate (Skin Cancer)",
        "causes": ["Long-term UV radiation exposure", "Fair skin type", "History of sunburns"],
        "symptoms": ["Pearly or waxy bump", "Flat flesh-colored or brown scar-like lesion", "Bleeding sore that heals and returns"],
        "suggestions": ["⚠️ See a dermatologist for biopsy and removal", "Avoid direct sun exposure", "Use sunscreen daily"],
    },
    "Benign Keratosis": {
        "emoji": "🟤", "severity": "Mild (Non-cancerous)",
        "causes": ["Genetic predisposition", "Aging", "Sun exposure"],
        "symptoms": ["Waxy, slightly elevated scaly appearance", "Pasted-on look", "Light tan to black color"],
        "suggestions": ["Usually requires no treatment", "Consult doctor if irritated or bleeding", "Do not scratch or pick"],
    },
    "Eczema": {
        "emoji": "🟡", "severity": "Mild–Severe (Chronic)",
        "causes": ["Genetic predisposition", "Immune system overreaction", "Allergens: dust mites, pollen"],
        "symptoms": ["Intense itching", "Dry, cracked, or scaly patches", "Red-to-brownish-gray patches"],
        "suggestions": ["Moisturize at least twice daily", "Use mild unscented soaps", "Identify and avoid triggers"],
    },
    "Melanocytic": {
        "emoji": "⚫", "severity": "Mild-Moderate (Moles/Nevi)",
        "causes": ["Localized overgrowth of melanocytes", "Genetics", "Sun exposure"],
        "symptoms": ["Usually uniform in color", "Round or oval shape", "Flat or raised"],
        "suggestions": ["Monitor for changes using ABCDE rule", "Apply sunscreen", "Consult a dermatologist if changes occur"],
    },
    "Melanoma": {
        "emoji": "🔴", "severity": "Potentially Life-Threatening",
        "causes": ["Excessive UV radiation", "Genetic mutations in melanocytes", "Family history of melanoma"],
        "symptoms": ["Asymmetric mole or lesion", "Irregular ragged border", "Multiple colors in one lesion"],
        "suggestions": ["⚠️ See a dermatologist IMMEDIATELY", "Undergo full-body skin examination", "Apply broad-spectrum SPF 30+ daily"],
    },
    "Psoriasis": {
        "emoji": "🟡", "severity": "Moderate–Severe (Chronic)",
        "causes": ["Autoimmune condition", "Genetic predisposition", "Triggers: stress, infections, skin injury"],
        "symptoms": ["Thick red patches with silvery-white scales", "Dry cracked skin that may bleed", "Thickened or pitted nails"],
        "suggestions": ["Keep skin moisturized", "Use prescribed topical corticosteroids", "Avoid known triggers"],
    },
    "Seborrheic": {
        "emoji": "🟠", "severity": "Mild",
        "causes": ["Malassezia yeast overgrowth", "Excess oil production", "Stress and fatigue"],
        "symptoms": ["Flaky scales (dandruff)", "Greasy patches with white or yellow scales", "Mild itching"],
        "suggestions": ["Use medicated creams or shampoos", "Wash with zinc pyrithione or ketoconazole shampoo", "Manage stress"],
    },
    "Tinea Ringworms Candidiasis": {
        "emoji": "🟢", "severity": "Mild (Highly Contagious)",
        "causes": ["Fungal infection by dermatophytes or yeast", "Direct skin-to-skin contact", "Warm moist environments"],
        "symptoms": ["Ring-shaped rash with clear center", "Itchy red scaly skin", "Blisters or pustules"],
        "suggestions": ["Keep area clean and dry", "Use OTC antifungal creams or sprays", "Avoid sharing personal items"],
    },
    "Warts Molluscum": {
        "emoji": "🦠", "severity": "Mild (Contagious)",
        "causes": ["Human Papillomavirus (HPV)", "Molluscum contagiosum virus (MCV)", "Direct contact with infected skin"],
        "symptoms": ["Small rough hard bumps (warts)", "Small raised bumps with dimple in center (molluscum)", "Can appear anywhere"],
        "suggestions": ["Avoid scratching or picking", "Use OTC wart treatments", "Consult a doctor for professional removal"],
    },
}

_FALLBACK_INFO = {
    "emoji": "🔬", "severity": "Unknown",
    "causes": ["Information not available for this condition."],
    "symptoms": ["Please consult a dermatologist for proper diagnosis."],
    "suggestions": ["Consult a certified dermatologist immediately."],
}


def get_disease_info(prediction: str) -> dict:
    if prediction in DISEASE_INFO:
        return DISEASE_INFO[prediction]
    pred_lower = prediction.lower()
    for key, val in DISEASE_INFO.items():
        if key.lower() in pred_lower or pred_lower in key.lower():
            return val
    return _FALLBACK_INFO


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/debug")
def debug():
    from backend.model_loader import MODEL_PATH, HF_REPO_ID, HF_TOKEN
    return jsonify({
        "model_loaded":          model is not None,
        "model_status":          model_status,
        "model_path":            MODEL_PATH,
        "model_exists_on_disk":  os.path.exists(MODEL_PATH),
        "HF_REPO_ID":            HF_REPO_ID or "NOT SET",
        "HF_TOKEN_set":          HF_TOKEN is not None,
        "load_error":            load_error or "none",
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
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    allowed = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        return jsonify({"error": f"Unsupported file type: {ext}"}), 400

    try:
        image_bytes = file.read()

        save_path = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
        with open(save_path, "wb") as f:
            f.write(image_bytes)

        media_type     = "image/jpeg" if ext in {".jpg", ".jpeg"} else f"image/{ext.lstrip('.')}"
        image_data_url = f"data:{media_type};base64,{base64.b64encode(image_bytes).decode()}"

        result     = predict_image(save_path, model)
        prediction = result["prediction"]
        confidence = result["confidence"]
        top3       = result["top3"]
        info       = get_disease_info(prediction)

        return jsonify({
            "prediction": prediction,
            "confidence": confidence,
            "top3":       top3,
            "image":      image_data_url,
            "info":       info,
        })

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)