# Load model when server starts
from backend.model_loader import load_skin_model

model = load_skin_model()
import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from backend.model_loader import load_skin_model
from backend.predict import predict_image
model = load_skin_model()

app = Flask(__name__, template_folder="../templates", static_folder="../static")
CORS(app)


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "..", "static", "uploaded_images")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DISEASE_INFO = {
    "Acne": {
        "emoji": "😣",
        "severity": "Mild–Moderate",
        "causes": ["Excess oil production", "Clogged hair follicles", "Bacterial infection", "Hormonal changes"],
        "symptoms": ["Whiteheads", "Blackheads", "Pimples", "Cysts on face/back/chest"],
        "suggestions": ["Wash face twice daily", "Use non-comedogenic products", "Consult a dermatologist for severe cases", "Avoid touching your face"]
    },
    "Eczema": {
        "emoji": "🩹",
        "severity": "Moderate",
        "causes": ["Genetic factors", "Immune system issues", "Environmental triggers", "Dry skin"],
        "symptoms": ["Itchy skin", "Red patches", "Dry scaly skin", "Inflammation"],
        "suggestions": ["Moisturize regularly", "Avoid known triggers", "Use gentle soap", "See a dermatologist for prescribed creams"]
    },
    "Psoriasis": {
        "emoji": "🔴",
        "severity": "Moderate–Severe",
        "causes": ["Autoimmune condition", "Genetic predisposition", "Stress", "Infections"],
        "symptoms": ["Red scaly patches", "Dry cracked skin", "Itching and burning", "Thickened nails"],
        "suggestions": ["Use prescribed topical treatments", "Avoid stress", "Keep skin moisturized", "Consult a dermatologist"]
    },
    "Melanoma": {
        "emoji": "⚠️",
        "severity": "Severe",
        "causes": ["UV radiation exposure", "Genetic mutations", "Fair skin type", "Family history"],
        "symptoms": ["Asymmetric mole", "Irregular borders", "Multiple colors", "Large diameter spot"],
        "suggestions": ["See a dermatologist immediately", "Avoid sun exposure", "Use SPF 50+ sunscreen", "Regular skin checks"]
    },
    "Basal Cell Carcinoma": {
        "emoji": "🔬",
        "severity": "Severe",
        "causes": ["Long-term UV exposure", "Radiation therapy", "Fair skin", "Weakened immune system"],
        "symptoms": ["Pearly bump on skin", "Flat flesh-colored lesion", "Bleeding sore", "Scar-like lesion"],
        "suggestions": ["Consult a dermatologist immediately", "Surgical removal may be needed", "Protect skin from sun", "Regular follow-ups"]
    },
    "Tinea Ringworm": {
        "emoji": "🟤",
        "severity": "Mild",
        "causes": ["Fungal infection", "Contact with infected person", "Sharing personal items", "Warm moist environment"],
        "symptoms": ["Ring-shaped rash", "Scaly itchy skin", "Red border", "Hair loss in affected area"],
        "suggestions": ["Use antifungal cream", "Keep skin dry and clean", "Avoid sharing personal items", "Wash clothes frequently"]
    },
    "Vitiligo": {
        "emoji": "⬜",
        "severity": "Mild",
        "causes": ["Autoimmune condition", "Genetic factors", "Stress", "Skin trauma"],
        "symptoms": ["White patches on skin", "Premature whitening of hair", "Loss of skin color", "Symmetrical patches"],
        "suggestions": ["Use sunscreen on affected areas", "Consult a dermatologist", "Camouflage cosmetics available", "Light therapy may help"]
    },
    "Seborrheic Keratosis": {
        "emoji": "🟫",
        "severity": "Mild",
        "causes": ["Aging", "Genetic factors", "Sun exposure", "Hormonal changes"],
        "symptoms": ["Waxy brown growths", "Rough scaly texture", "Oval shaped patches", "Itching in some cases"],
        "suggestions": ["Usually harmless, no treatment needed", "Removal if irritated", "Consult dermatologist if changing", "Monitor for any changes"]
    }
}

def get_disease_info(prediction):
    for key in DISEASE_INFO:
        if key.lower() in prediction.lower() or prediction.lower() in key.lower():
            return DISEASE_INFO[key]
    return {
        "emoji": "🔬",
        "severity": "Unknown",
        "causes": ["Consult a dermatologist for accurate diagnosis"],
        "symptoms": ["Please see a medical professional"],
        "suggestions": ["Visit a qualified dermatologist"]
    }

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['image']

        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        result = predict_image(file_path)

        # Handle if predict_image returns a dict or a string
        if isinstance(result, dict):
            prediction = result.get("prediction", "Unknown")
            confidence = result.get("confidence", 0)
            top3 = result.get("top3", [{"label": prediction, "confidence": confidence}])
        else:
            prediction = str(result)
            confidence = 0
            top3 = [{"label": prediction, "confidence": 0}]

        info = get_disease_info(prediction)

        return jsonify({
            "prediction": prediction,
            "confidence": confidence,
            "top3": top3,
            "info": info
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)