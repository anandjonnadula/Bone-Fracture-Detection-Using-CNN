from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2

def build_cnn_model(input_shape=(224, 224, 3), num_classes=1):
    """
    Build a simple CNN model.
    - num_classes=1 -> binary classification
    - num_classes>1 -> multi-class classification
    """
    model = Sequential()

    # 1st Convolution Block
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # 2nd Convolution Block
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # 3rd Convolution Block
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten
    model.add(Flatten())

    # Fully Connected Layers
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    # Output Layer
    if num_classes == 1:
        model.add(Dense(1, activation='sigmoid'))  # Binary
        loss_fn = 'binary_crossentropy'
    else:
        model.add(Dense(num_classes, activation='softmax'))  # Multi-class
        loss_fn = 'categorical_crossentropy'

    # Compile Model
    model.compile(
        optimizer='adam',
        loss=loss_fn,
        metrics=['accuracy']
    )

    return model

def build_transfer_learning_model(input_shape=(224, 224, 3), num_classes=12):
    """
    Build a more accurate model using Transfer Learning (MobileNetV2).
    """
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze base layers initially

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model