from backend.model_loader import load_skin_model
import numpy as np
from tensorflow.keras.preprocessing import image

# Load model once
model = load_skin_model()

def predict_image(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        preds = model.predict(img_array)

        predicted_class = np.argmax(preds)
        confidence = float(np.max(preds))

        return {
            "prediction": str(predicted_class),
            "confidence": confidence
        }

    except Exception as e:
        return {"error": str(e)}