import os
import tensorflow as tf

MODEL_PATH = "/opt/render/project/src/models/skin_model.h5"

model = None

def load_model():
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            print("Downloading model...")
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            import gdown
            gdown.download(
                "https://drive.google.com/uc?id=1S2tDM5qMhqnDgx7fMK4o5aq2uzQxFfjn",
                MODEL_PATH, quiet=False
            )
        model = tf.keras.models.load_model(MODEL_PATH)
    return model