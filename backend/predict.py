import numpy as np
from backend.model_loader import load_model
from utils.preprocess import preprocess_image
from utils.labels import labels

def predict_image(image_path):
    model = load_model()
    processed = preprocess_image(image_path)
    predictions = model.predict(processed)[0]

    top_indices = np.argsort(predictions)[::-1][:3]

    top3 = [
        {
            "label": labels[i],
            "confidence": round(float(predictions[i]) * 100, 2)
        }
        for i in top_indices
    ]

    return {
        "prediction": top3[0]["label"],
        "confidence": top3[0]["confidence"],
        "top3": top3
    }