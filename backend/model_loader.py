import os
from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model

# Hugging Face settings
HF_REPO_ID = "tusharburai/Skin-Disease-Detection"
HF_FILENAME = "skin_model.h5"

# Optional: for private repositories
HF_TOKEN = os.getenv("HF_TOKEN")

_model = None


def load_skin_model():
    global _model

    if _model is not None:
        return _model

    try:
        print("⬇️ Downloading model from Hugging Face...")

        model_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_FILENAME,
            token=HF_TOKEN if HF_TOKEN else None
        )

        print(f"✅ Model downloaded to: {model_path}")

        _model = load_model(model_path)

        print("✅ Model loaded successfully")
        print(f"📊 Output shape: {_model.output_shape}")

        return _model

    except Exception as e:
        print(f"❌ Model loading failed: {str(e)}")
        return None