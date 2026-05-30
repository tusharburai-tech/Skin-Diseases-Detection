import os
import gdown
from tensorflow.keras.models import load_model

# Resolve to project ROOT (one level above /backend/)
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR  = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "skin_model.h5")

FILE_ID = "1S2tDM5qMhqnDgx7fMK4o5aq2uzQxFfjn"

os.makedirs(MODEL_DIR, exist_ok=True)

_model = None  # module-level cache — loads only once

def load_skin_model():
    global _model
    if _model is not None:
        return _model  # already loaded, reuse

    if not os.path.exists(MODEL_PATH):
        print("⬇️  Model not found — downloading from Google Drive...")
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        try:
            gdown.download(url, MODEL_PATH, quiet=False)
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return None

    print(f"📁 Model path : {MODEL_PATH}")
    print(f"✅ File exists : {os.path.exists(MODEL_PATH)}")

    try:
        _model = load_model(MODEL_PATH)
        print(f"✅ Model loaded — output classes: {_model.output_shape[-1]}")
        return _model
    except Exception as e:
        print(f"⚠️  Could not load model: {e}")
        return None