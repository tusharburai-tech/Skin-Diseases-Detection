import os
import shutil
from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model

HF_REPO_ID  = os.environ.get("HF_REPO_ID", "tusharburai/Skin-Disease-Detection")
HF_FILENAME = "skin_model.h5"
HF_TOKEN    = os.environ.get("HF_TOKEN", None)

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR  = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "skin_model.h5")

os.makedirs(MODEL_DIR, exist_ok=True)

_model = None

def load_skin_model():
    global _model
    if _model is not None:
        return _model
    if not os.path.exists(MODEL_PATH):
        try:
            print(f"⬇️  Downloading from Hugging Face: {HF_REPO_ID}")
            downloaded = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=HF_FILENAME,
                token=HF_TOKEN,
            )
            shutil.copy(downloaded, MODEL_PATH)
            print(f"✅ Model saved to: {MODEL_PATH}")
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return None
    try:
        _model = load_model(MODEL_PATH)
        print(f"✅ Model loaded — classes: {_model.output_shape[-1]}")
        return _model
    except Exception as e:
        print(f"❌ Could not load model: {e}")
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        return None
