import os
import gdown
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "skin_model.h5")

# Create models folder
os.makedirs(MODEL_DIR, exist_ok=True)


# Download model if not exists
if not os.path.exists(MODEL_PATH):
    print("⬇️ Downloading model from Google Drive...")
    url = f"https://drive.google.com/file/d/1S2tDM5qMhqnDgx7fMK4o5aq2uzQxFfjn/view?usp=drive_link"
    gdown.download(url, MODEL_PATH, quiet=False)

# Load model
model = None
try:
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
        print("✅ Model loaded successfully")
    else:
        print("❌ Model file not found after download")
except Exception as e:
    print("❌ Error loading model:", e)