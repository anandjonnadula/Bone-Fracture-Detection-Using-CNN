import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os
import json
from PIL import ImageFile
from cnn_model import build_transfer_learning_model

# ✅ Fix truncated/corrupted image errors
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.join(BASE_DIR, "../dataset/stage2_MultiClassification/train")
test_dir = os.path.join(BASE_DIR, "../dataset/stage2_MultiClassification/test")

# Data Generators - enhanced augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Print and save class indices
print("Class Indices:", train_data.class_indices)
with open(os.path.join(BASE_DIR, "class_indices.json"), "w") as f:
    json.dump(train_data.class_indices, f)

# Build Transfer Learning Model
model = build_transfer_learning_model(num_classes=train_data.num_classes)

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001)

# Train the Model
model.fit(
    train_data,
    validation_data=test_data,
    epochs=50,  # Increased epochs
    callbacks=[early_stop, lr_reduce],
    verbose=1
)

# Optional: Fine-tuning (unfreeze base model)
print("Starting fine-tuning...")
model.layers[0].trainable = True
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),  # Very low learning rate
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    train_data,
    validation_data=test_data,
    epochs=30,
    callbacks=[early_stop, lr_reduce],
    verbose=1
)

# Ensure saved_model folder exists
save_path = os.path.join(BASE_DIR, "saved_model")
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Save Stage 2 Model
model.save(os.path.join(save_path, "stage2_model.h5"))
print("✅ Stage 2 Model Saved Successfully!")
