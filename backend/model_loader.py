import os
import gdown
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "skin_model.h5")
print("Model path:", MODEL_PATH)
print("Exists before load:", os.path.exists(MODEL_PATH))
os.makedirs(MODEL_DIR, exist_ok=True)

FILE_ID = "1S2tDM5qMhqnDgx7fMK4o5aq2uzQxFfjn"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("⬇️ Downloading model...")
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
    else:
        print("✅ Model already exists")

def load_skin_model():
    download_model()
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model still not found after download!")

    print("📦 Loading model...")
    return load_model(MODEL_PATH)