import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import os
from PIL import ImageFile

# ✅ Fix truncated image errors
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Paths - make sure these match your dataset structure
train_dir = "../dataset/stage1_BinaryClassification/train"
test_dir = "../dataset/stage1_BinaryClassification/test"

# Data Generators - include basic data augmentation for robustness
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=True
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# Simple CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),   # ✅ Add dropout to avoid overfitting
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

# Compile Model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train the Model
model.fit(
    train_data,
    validation_data=test_data,
    epochs=15,   # ✅ Increase epochs for better learning
    verbose=1
)

# Ensure saved_model folder exists
save_path = os.path.join(os.getcwd(), "saved_model")
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Save model
model.save(os.path.join(save_path, "stage1_model.h5"))
print("✅ Stage 1 Model Saved Successfully!")