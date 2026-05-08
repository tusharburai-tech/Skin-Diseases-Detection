import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
import io
import base64

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = os.path.join("static", "uploaded_images")
MODEL_PATH    = os.path.join("models", "skin_model.h5")
IMG_SIZE      = (224, 224)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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
    "Warts Molluscum"
]

DISEASE_INFO = {
    "Acne": {
        "emoji": "🔴",
        "severity": "Mild–Moderate",
        "causes": [
            "Excess sebum (oil) production clogging hair follicles",
            "Bacterial overgrowth (Cutibacterium acnes)",
            "Hormonal changes during puberty, menstruation, or stress",
            "Diet high in refined carbs, dairy, or sugar",
            "Certain medications (corticosteroids, lithium)",
        ],
        "symptoms": [
            "Whiteheads and blackheads",
            "Papules, pustules, or cysts",
            "Oily skin, especially on the face, chest, or back",
            "Scarring after severe breakouts",
        ],
        "suggestions": [
            "Cleanse gently twice daily with a mild, non-comedogenic cleanser",
            "Use topical benzoyl peroxide or salicylic acid products",
            "Avoid touching, picking, or popping pimples",
            "Maintain a low-glycemic diet and drink plenty of water",
            "Consult a dermatologist for prescription retinoids or antibiotics if severe",
        ],
    },
    "Eczema": {
        "emoji": "🟡",
        "severity": "Mild–Severe (Chronic)",
        "causes": [
            "Genetic predisposition affecting skin barrier proteins",
            "Immune system overreaction to environmental triggers",
            "Allergens: dust mites, pet dander, pollen, mold",
            "Irritants: soaps, detergents, synthetic fabrics",
            "Stress and extreme temperature changes",
        ],
        "symptoms": [
            "Intense itching, especially at night",
            "Dry, cracked, or scaly patches of skin",
            "Red-to-brownish-gray patches",
            "Small raised bumps that may weep fluid when scratched",
            "Thickened, leathery skin from chronic scratching",
        ],
        "suggestions": [
            "Moisturize at least twice daily with thick, fragrance-free creams",
            "Use mild, unscented soaps and detergents",
            "Identify and avoid personal triggers",
            "Apply prescribed topical corticosteroids during flare-ups",
            "Wear soft, breathable cotton clothing",
            "Keep nails short to minimize skin damage from scratching",
        ],
    },
    "Herpes Zoster": {
        "emoji": "🟠",
        "severity": "Moderate–Severe",
        "causes": [
            "Reactivation of the Varicella-Zoster Virus (VZV), which causes chickenpox",
            "Virus lies dormant in nerve tissue and reactivates when immunity weakens",
            "Risk increases with age (most common after 50)",
            "Immunosuppression from illness, stress, or medications",
        ],
        "symptoms": [
            "Burning, shooting pain or tingling before rash appears",
            "Stripe of blisters wrapping around one side of torso, face, or eye",
            "Fluid-filled blisters that break open and crust over",
            "Fever, headache, and fatigue",
            "Post-herpetic neuralgia (persistent nerve pain) in some cases",
        ],
        "suggestions": [
            "Seek antiviral treatment (acyclovir, valacyclovir) within 72 hours of rash onset",
            "Keep the rash clean and dry to prevent bacterial infection",
            "Use calamine lotion or cool compresses to soothe itching",
            "Take OTC pain relievers; consult a doctor for severe pain",
            "Get the Shingrix vaccine if you are 50+ and haven't been vaccinated",
            "Avoid contact with immunocompromised individuals or newborns while blisters are active",
        ],
    },
    "Melanoma": {
        "emoji": "🔴",
        "severity": "Potentially Life-Threatening",
        "causes": [
            "Excessive UV radiation from sun exposure or tanning beds",
            "Genetic mutations in melanocytes (skin pigment cells)",
            "Family history of melanoma or atypical moles",
            "Fair skin, light eyes, and red or blonde hair",
            "History of severe sunburns, especially in childhood",
        ],
        "symptoms": [
            "Asymmetric mole or lesion (one half doesn't match the other)",
            "Irregular, ragged, or blurred border",
            "Multiple colors in one lesion (brown, black, red, white, or blue)",
            "Diameter larger than 6mm (pencil eraser size)",
            "Evolving size, shape, color, or new symptom like bleeding",
        ],
        "suggestions": [
            "⚠️ See a dermatologist IMMEDIATELY — early detection saves lives",
            "Do not attempt any home treatment",
            "Undergo a full-body skin examination by a specialist",
            "Apply broad-spectrum SPF 30+ sunscreen daily, even on cloudy days",
            "Wear protective clothing, hats, and UV-blocking sunglasses",
            "Perform monthly self-exams and watch for any ABCDE changes",
        ],
    },
    "Psoriasis": {
        "emoji": "🟡",
        "severity": "Moderate–Severe (Chronic)",
        "causes": [
            "Autoimmune condition causing rapid skin cell turnover",
            "Genetic predisposition (family history is a strong factor)",
            "Triggers: stress, infections, injury to skin, smoking, heavy alcohol use",
            "Certain medications: beta-blockers, lithium, antimalarials",
        ],
        "symptoms": [
            "Thick, red patches covered with silvery-white scales",
            "Dry, cracked skin that may bleed",
            "Itching, burning, or soreness around patches",
            "Thickened, pitted, or ridged nails",
            "Swollen and stiff joints (psoriatic arthritis in ~30% of cases)",
        ],
        "suggestions": [
            "Keep skin moisturized with heavy creams or ointments",
            "Use prescribed topical corticosteroids or vitamin D analogues",
            "Consider light therapy (phototherapy) under medical supervision",
            "Avoid known triggers: stress, alcohol, smoking, and skin injuries",
            "Ask your doctor about systemic treatments or biologics for severe cases",
            "Join a support group — psoriasis is a lifelong condition",
        ],
    },
    "Ringworm": {
        "emoji": "🟢",
        "severity": "Mild (Highly Contagious)",
        "causes": [
            "Fungal infection caused by dermatophytes (not an actual worm)",
            "Direct skin-to-skin contact with infected person or animal",
            "Contact with contaminated surfaces, clothing, or towels",
            "Warm, moist environments (gyms, locker rooms, swimming pools)",
            "Weakened immune system or diabetes",
        ],
        "symptoms": [
            "Ring-shaped rash with a clear center and raised edges",
            "Scaly, itchy, red or silver patches",
            "Gradually expanding rings",
            "Blisters or pustules at the edge of the ring",
            "Hair loss in affected areas (on scalp)",
        ],
        "suggestions": [
            "Apply OTC antifungal cream, powder, or spray (clotrimazole, miconazole) for 2–4 weeks",
            "Keep the affected area clean and dry at all times",
            "Avoid sharing personal items like towels, combs, and clothing",
            "Wash hands thoroughly after touching pets",
            "Consult a doctor for scalp ringworm — oral antifungals are required",
            "Continue treatment for the full prescribed period even after symptoms clear",
        ],
    },
    "Rosacea": {
        "emoji": "🟠",
        "severity": "Mild–Moderate (Chronic)",
        "causes": [
            "Exact cause unknown; combination of genetic and environmental factors",
            "Abnormal immune response or neurovascular dysregulation",
            "Triggers: sun exposure, spicy foods, alcohol, hot beverages",
            "Demodex mite overgrowth on the skin",
            "Helicobacter pylori bacterial infection (linked in some studies)",
        ],
        "symptoms": [
            "Persistent facial redness, especially on cheeks, nose, chin, and forehead",
            "Visible blood vessels (telangiectasia)",
            "Acne-like bumps and pimples",
            "Thickened, bumpy skin texture (rhinophyma) on nose in severe cases",
            "Eye irritation and redness (ocular rosacea)",
        ],
        "suggestions": [
            "Use gentle, fragrance-free skincare products",
            "Apply broad-spectrum SPF 30+ sunscreen every morning",
            "Identify and avoid personal triggers (keep a trigger diary)",
            "Use prescribed topical metronidazole or azelaic acid",
            "Consider laser therapy to reduce visible blood vessels",
            "Avoid extreme temperatures, strenuous exercise in heat, and alcohol",
        ],
    },
    "Vitiligo": {
        "emoji": "⚪",
        "severity": "Mild (Non-Contagious)",
        "causes": [
            "Autoimmune destruction of melanocytes (pigment-producing cells)",
            "Genetic predisposition with multiple gene variants involved",
            "Triggers: severe sunburn, skin trauma, emotional stress",
            "Associated with other autoimmune conditions (thyroid disease, diabetes)",
        ],
        "symptoms": [
            "Milky-white patches on skin, often starting on hands, face, or around body openings",
            "Premature whitening of hair, eyelashes, or eyebrows",
            "Loss of color inside the mouth or nose",
            "Patches that expand over time and can cover large areas",
        ],
        "suggestions": [
            "Apply SPF 30+ sunscreen on depigmented areas — they burn easily",
            "Consult a dermatologist for topical calcineurin inhibitors or corticosteroids",
            "Ask about narrowband UVB phototherapy for repigmentation",
            "Use cosmetic camouflage (self-tanners, makeup) to even out skin tone",
            "Explore the newer JAK inhibitor (ruxolitinib cream) — approved for vitiligo",
            "Connect with support communities; vitiligo is not contagious and not life-threatening",
        ],
    },
}

DISEASE_INFO.update({
    "Atopic Dermatitis": {
        "emoji": "🟡",
        "severity": "Mild–Severe (Chronic)",
        "causes": ["Genetic mutations affecting skin barrier", "Immune system overreaction", "Environmental triggers like dust mites or pollen"],
        "symptoms": ["Dry, cracked skin", "Itchiness (pruritus)", "Red to brownish-gray patches"],
        "suggestions": ["Moisturize frequently", "Avoid harsh soaps and irritants", "Use prescribed topical corticosteroids"]
    },
    "Basal Cell": {
        "emoji": "🔴",
        "severity": "Moderate (Skin Cancer)",
        "causes": ["Long-term exposure to ultraviolet (UV) radiation", "Fair skin type", "History of sunburns"],
        "symptoms": ["Pearly or waxy bump", "Flat, flesh-colored or brown scar-like lesion", "Bleeding or scabbing sore that heals and returns"],
        "suggestions": ["⚠️ See a dermatologist for biopsy and removal", "Avoid direct sun exposure", "Use sunscreen daily"]
    },
    "Benign Keratosis": {
        "emoji": "🟤",
        "severity": "Mild (Non-cancerous)",
        "causes": ["Genetic predisposition", "Aging (very common in older adults)", "Sun exposure in some cases"],
        "symptoms": ["Waxy, slightly elevated scaly appearance", "Pasted-on look", "Light tan to black color"],
        "suggestions": ["Usually requires no treatment", "Consult a doctor if it gets irritated or bleeds", "Do not scratch or pick at it"]
    },
    "Melanocytic": {
        "emoji": "⚫",
        "severity": "Mild-Moderate (Moles/Nevi)",
        "causes": ["Localized overgrowth of melanocytes", "Genetics", "Sun exposure"],
        "symptoms": ["Usually uniform in color", "Round or oval shape", "Flat or raised from the skin"],
        "suggestions": ["Monitor for changes in size, shape, or color (ABCDE rule)", "Apply sunscreen", "Consult a dermatologist if changes occur"]
    },
    "Seborrheic": {
        "emoji": "🟠",
        "severity": "Mild",
        "causes": ["Malassezia yeast overgrowth", "Excess oil production", "Stress and fatigue"],
        "symptoms": ["Flaky scales (dandruff)", "Greasy patches of skin covered with white or yellow scales", "Redness or mild itching"],
        "suggestions": ["Use over-the-counter medicated creams", "Wash affected area with zinc pyrithione or ketoconazole shampoo", "Manage stress levels"]
    },
    "Tinea Ringworms Candidiasis": {
        "emoji": "🟢",
        "severity": "Mild (Highly Contagious)",
        "causes": ["Fungal infection caused by dermatophytes or yeast", "Direct skin-to-skin contact", "Warm, moist environments"],
        "symptoms": ["Ring-shaped rash with clear center", "Itchy, red, scaly, or cracked skin", "Blisters or pustules"],
        "suggestions": ["Keep the affected area clean and dry", "Use OTC antifungal creams or sprays", "Avoid sharing personal items"]
    },
    "Warts Molluscum": {
        "emoji": "🦠",
        "severity": "Mild (Contagious)",
        "causes": ["Human Papillomavirus (HPV) for Warts", "Molluscum contagiosum virus (MCV)", "Direct contact with infected skin or surfaces"],
        "symptoms": ["Small, rough, hard bumps (warts)", "Small, raised, firm, and painless bumps with a dimple in the center (molluscum)", "Can appear anywhere on the body"],
        "suggestions": ["Avoid scratching or picking to prevent spreading", "Use OTC wart treatments", "Consult a doctor for cryotherapy or professional removal"]
    }
})

model = None

def load_model():
    global model
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"✅ Model loaded — output classes: {model.output_shape[-1]}")
    except Exception as e:
        print(f"⚠️  Could not load model: {e}")
        model = None

load_model()


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    allowed = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        return jsonify({"error": f"Unsupported file type: {ext}"}), 400

    image_bytes = file.read()

    save_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(save_path, "wb") as f:
        f.write(image_bytes)

    media_type = "image/jpeg" if ext in {".jpg", ".jpeg"} else f"image/{ext.lstrip('.')}"
    b64_image  = base64.b64encode(image_bytes).decode("utf-8")
    image_data_url = f"data:{media_type};base64,{b64_image}"

    if model is None:
        return jsonify({
            "error": "Model not loaded. Check that models/skin_model.h5 exists.",
            "image": image_data_url,
        }), 500

    try:
        input_arr  = preprocess_image(image_bytes)
        preds      = model.predict(input_arr)[0]

        num_classes = len(preds)
        if len(CLASS_NAMES) != num_classes:
            labels = (
                CLASS_NAMES[:num_classes]
                if len(CLASS_NAMES) > num_classes
                else CLASS_NAMES + [f"Class_{i}" for i in range(len(CLASS_NAMES), num_classes)]
            )
        else:
            labels = CLASS_NAMES

        top_idx    = int(np.argmax(preds))
        confidence = float(preds[top_idx]) * 100
        prediction = labels[top_idx]

        top3_idx = np.argsort(preds)[::-1][:3]
        top3 = [
            {"label": labels[i], "confidence": round(float(preds[i]) * 100, 2)}
            for i in top3_idx
        ]

        info = DISEASE_INFO.get(prediction, {
            "emoji": "🔬",
            "severity": "Unknown",
            "causes": ["Information not available for this condition."],
            "symptoms": ["Please consult a dermatologist for proper diagnosis."],
            "suggestions": ["Consult a certified dermatologist immediately."],
        })

        return jsonify({
            "prediction": prediction,
            "confidence": round(confidence, 2),
            "top3":       top3,
            "image":      image_data_url,
            "info":       info,
        })

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)