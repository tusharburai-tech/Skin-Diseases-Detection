import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# 🔧 Config
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10

train_dir = "dataset/train"

# 📊 Data Generator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# 📌 Print class labels
print("Classes:", train_data.class_indices)

# 🧠 Model (CNN)
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(train_data.num_classes, activation='softmax')
])

# ⚙️ Compile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 🚀 Train
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# 💾 Save model
os.makedirs("models", exist_ok=True)
model.save("models/skin_model.h5")

print("✅ Model trained and saved!")