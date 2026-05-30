from backend.model_loader import load_skin_model
import numpy as np
from tensorflow.keras.preprocessing import image

model = load_skin_model()

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)

    return {
        "prediction": str(np.argmax(preds)),
        "confidence": float(np.max(preds))
    }