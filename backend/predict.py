import numpy as np
from PIL import Image

CLASS_NAMES = [
    "Atopic Dermatitis",
    "Basal Cell",
    "Benign Keratosis",
    "Eczema",
    "Melanocytic",
    "Melanoma",
    "Psoriasis",
    "Seborrheic",
    "Tinea Ringworms Candidiasis",
    "Warts Molluscum",
]

IMG_SIZE = (224, 224)


def predict_image(img_path, model):
    """Run inference on a saved image file.

    Args:
        img_path : path to the image on disk
        model    : loaded Keras model (passed in — NOT loaded here)

    Returns:
        dict with prediction (str), confidence (float %), top3 list
    """
    img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr)[0]

    num_classes = len(preds)
    if len(CLASS_NAMES) >= num_classes:
        labels = CLASS_NAMES[:num_classes]
    else:
        labels = CLASS_NAMES + [
            f"Class_{i}" for i in range(len(CLASS_NAMES), num_classes)
        ]

    top_idx    = int(np.argmax(preds))
    confidence = round(float(preds[top_idx]) * 100, 2)
    prediction = labels[top_idx]

    top3_idx = np.argsort(preds)[::-1][:3]
    top3 = [
        {"label": labels[i], "confidence": round(float(preds[i]) * 100, 2)}
        for i in top3_idx
    ]

    return {
        "prediction": prediction,
        "confidence": confidence,
        "top3":       top3,
    }