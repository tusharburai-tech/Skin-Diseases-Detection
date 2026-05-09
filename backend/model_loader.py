import os
import tensorflow as tf

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "models", "skin_model.h5")

model = None

def load_model():
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            print("Downloading model from Google Drive...")
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            import gdown
            FILE_ID = "1S2tDM5qMhqnDgx7fMK4o5aq2uzQxFfjn"
            gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH, quiet=False)
        model = tf.keras.models.load_model(MODEL_PATH)
    return model
