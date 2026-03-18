# src/model.py

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB3
import logging

log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
IMG_SIZE     = 224
DROPOUT_RATE = 0.4


def build_model(freeze_backbone: bool = True) -> Model:
    """
    Builds EfficientNetB3 regression model for AQI prediction.

    Architecture:
        Input (224x224x3)
            → EfficientNetB3 backbone (pretrained ImageNet)
            → GlobalAveragePooling2D
            → Dense(256, swish) + BatchNorm + Dropout
            → Dense(128, swish) + BatchNorm + Dropout
            → Dense(1, linear)  ← normalized AQI output [0, 1]

    Args:
        freeze_backbone: If True, freezes all EfficientNetB3 weights.
                         Set False during fine-tuning stage.
    Returns:
        Compiled Keras Model
    """

    # ── Input ──────────────────────────────────────────────────────────────────
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input_image")

    # ── Backbone ───────────────────────────────────────────────────────────────
    backbone = EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs
    )
    backbone.trainable = not freeze_backbone

    if freeze_backbone:
        log.info(f"Backbone frozen — {count_frozen(backbone)} layers locked")
    else:
        log.info(f"Backbone unfrozen — fine-tuning all {len(backbone.layers)} layers")

    x = backbone.output

    # ── Regression Head ────────────────────────────────────────────────────────
    x = layers.GlobalAveragePooling2D(name="gap")(x)

    x = layers.Dense(256, name="dense_256")(x)
    x = layers.Activation("swish", name="swish_256")(x)
    x = layers.BatchNormalization(name="bn_256")(x)
    x = layers.Dropout(DROPOUT_RATE, name="drop_256")(x)

    x = layers.Dense(128, name="dense_128")(x)
    x = layers.Activation("swish", name="swish_128")(x)
    x = layers.BatchNormalization(name="bn_128")(x)
    x = layers.Dropout(DROPOUT_RATE / 2, name="drop_128")(x)

    # Linear output — predicts normalized AQI in [0, 1]
    output = layers.Dense(1, activation="linear", name="aqi_output")(x)

    # ── Compile ────────────────────────────────────────────────────────────────
    model = Model(inputs=inputs, outputs=output, name="AQI_EfficientNetB3")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.Huber(delta=0.5),
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            tf.keras.metrics.RootMeanSquaredError(name="rmse")
        ]
    )

    return model


def unfreeze_backbone(model: Model, learning_rate: float = 1e-5,
                      unfreeze_from_percent: float = 0.7) -> Model:
    """Unfreezes only the top 30% of backbone layers."""
    for layer in model.layers:
        if "efficientnetb3" in layer.name:
            total        = len(layer.layers)
            freeze_until = int(total * unfreeze_from_percent)
            for i, l in enumerate(layer.layers):
                l.trainable = i >= freeze_until
            log.info(f"Unfreezing top {total - freeze_until}/{total} backbone layers")

    # Keep head always trainable
    for layer in model.layers:
        if "efficientnetb3" not in layer.name:
            layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.Huber(delta=0.5),
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            tf.keras.metrics.RootMeanSquaredError(name="rmse")
        ]
    )
    return model

def count_frozen(backbone) -> int:
    """Returns number of non-trainable layers in backbone."""
    return sum(1 for l in backbone.layers if not l.trainable)


def model_summary(model: Model):
    """Prints a clean summary with parameter counts."""
    total     = model.count_params()
    trainable = sum(tf.size(w).numpy() for w in model.trainable_weights)
    frozen    = total - trainable

    print("\n" + "═" * 55)
    print(f"  Model      : {model.name}")
    print(f"  Input      : {model.input_shape}")
    print(f"  Output     : {model.output_shape}")
    print(f"  Total params     : {total:>12,}")
    print(f"  Trainable        : {trainable:>12,}")
    print(f"  Frozen           : {frozen:>12,}")
    print("═" * 55 + "\n")


# ── Entry Point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    print("\n── Stage 1: Frozen backbone ──")
    model_frozen = build_model(freeze_backbone=True)
    model_summary(model_frozen)

    print("\n── Stage 2: Unfrozen backbone ──")
    model_ft = unfreeze_backbone(model_frozen, learning_rate=1e-5)
    model_summary(model_ft)