"""Model architectures for the two-stage bone fracture detection pipeline.

Both stages use MobileNetV2 transfer learning. The backbone is grafted into
the model graph via `input_tensor` (instead of nesting it as a single layer)
so that Grad-CAM can reach the internal conv layers of the final model
without running into disconnected-graph errors.

Pixel preprocessing (MobileNetV2 expects inputs in [-1, 1]) is baked into the
model as a Rescaling layer, so callers always feed raw 0-255 RGB pixels. This
removes a whole class of train/inference preprocessing mismatches.
"""

from tensorflow.keras import Input, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    BatchNormalization,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Rescaling,
)

IMG_SIZE = 224
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

# Name of the last activation after the final conv block of MobileNetV2 —
# the layer Grad-CAM visualizes.
LAST_CONV_LAYER = "out_relu"


def _build_backbone(inputs):
    """Graft an ImageNet-pretrained MobileNetV2 onto `inputs` (raw 0-255)."""
    x = Rescaling(1.0 / 127.5, offset=-1.0, name="preprocess_rescale")(inputs)
    base = MobileNetV2(input_tensor=x, include_top=False, weights="imagenet")
    return base


def freeze_backbone(base):
    for layer in base.layers:
        layer.trainable = False


def unfreeze_top_layers(base, n_layers=40):
    """Unfreeze the last `n_layers` of the backbone for fine-tuning.

    BatchNormalization layers stay frozen: updating their statistics with
    small medical datasets destroys the pretrained ImageNet statistics and
    reliably hurts accuracy.
    """
    for layer in base.layers:
        layer.trainable = False
    for layer in base.layers[-n_layers:]:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = True


def build_stage1_model(input_shape=INPUT_SHAPE):
    """Binary fracture / no-fracture classifier.

    Output: sigmoid P(fracture) — the positive class is 'fracture'.
    Returns (model, backbone) so training code can fine-tune the backbone.
    """
    inputs = Input(shape=input_shape, name="xray_input")
    base = _build_backbone(inputs)
    freeze_backbone(base)

    x = GlobalAveragePooling2D(name="gap")(base.output)
    x = Dropout(0.3, name="head_dropout")(x)
    outputs = Dense(1, activation="sigmoid", name="fracture_prob")(x)

    model = Model(inputs, outputs, name="stage1_fracture_detector")
    return model, base


def build_stage2_model(num_classes, input_shape=INPUT_SHAPE):
    """12-way fracture-type classifier.

    Returns (model, backbone) so training code can fine-tune the backbone.
    """
    inputs = Input(shape=input_shape, name="xray_input")
    base = _build_backbone(inputs)
    freeze_backbone(base)

    x = GlobalAveragePooling2D(name="gap")(base.output)
    x = Dropout(0.35, name="head_dropout")(x)
    outputs = Dense(num_classes, activation="softmax", name="fracture_type")(x)

    model = Model(inputs, outputs, name="stage2_fracture_classifier")
    return model, base
