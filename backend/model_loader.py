import os
from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR  = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "skin_model.h5")

# 👇 Replace with YOUR actual Hugging Face username and repo name
HF_REPO_ID  = "tusharburai/Skin-Disease-Detection"
HF_FILENAME = "skin_model.h5"

os.makedirs(MODEL_DIR, exist_ok=True)

_model = None

def load_skin_model():
    global _model   
    if _model is not None:
        return _model

    if not os.path.exists(MODEL_PATH):
        print("⬇️  Downloading model from Hugging Face...")
        try:
            downloaded_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=HF_FILENAME,
                local_dir=MODEL_DIR,
            )
            print(f"✅ Downloaded to: {downloaded_path}")
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return None

    try:
        _model = load_model(MODEL_PATH)
        print(f"✅ Model loaded — output classes: {_model.output_shape[-1]}")
        return _model
    except Exception as e:
        print(f"⚠️  Could not load model: {e}")
        return None