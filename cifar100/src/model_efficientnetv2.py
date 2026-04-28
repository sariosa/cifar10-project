"""
model_efficientnetv2.py

EfficientNetV2S transfer learning model for CIFAR-100.

Why EfficientNetV2S instead of EfficientNetB3?
-----------------------------------------------
EfficientNetV2 (Tan & Le, 2021) improves on EfficientNetB3 in three ways:
  1. Fused-MBConv blocks in early layers — faster training and inference.
  2. Progressive learning support — designed to handle variable input sizes.
  3. Better accuracy-vs-parameters trade-off on ImageNet.

EfficientNetV2S achieves ~84% top-1 on ImageNet vs ~81% for EfficientNetB3,
with roughly the same parameter count. On CIFAR-100 this typically translates
to 1–3% higher test accuracy.

Architecture of the custom head (same reasoning as transfer_learning.py):
  - GlobalAveragePooling2D   → reduce spatial dims without flattening
  - BatchNormalization        → stabilise activations from a new head
  - Dense(512, relu)          → task-specific feature combination
  - Dropout(0.5)              → prevent co-adaptation in the small head
  - Dense(100, softmax)       → 100-class output

How to use:
    from cifar100.src.model_efficientnetv2 import build_model, unfreeze_top_layers

    model, base = build_model(num_epochs_phase1=15)
    # ... train phase 1 ...
    unfreeze_top_layers(model, base, n_layers=50)
    # ... recompile and train phase 2 ...
"""

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras import layers


def build_model(
    input_shape: tuple = (224, 224, 3),
    num_classes: int = 100,
    dropout_rate: float = 0.5,
    initial_lr: float = 1e-3,
    num_epochs_phase1: int = 15,
    steps_per_epoch: int = 1407,     # ≈ 45000 // 32; recalculated in train_advanced.py
    label_smoothing: float = 0.1,
):
    """
    Build and compile the EfficientNetV2S transfer learning model.

    The base model is frozen initially so only the custom head is trained
    in Phase 1. Call unfreeze_top_layers() before Phase 2.

    Args:
        input_shape:       Model input dimensions. Default (224, 224, 3).
        num_classes:       Number of output classes. Default 100.
        dropout_rate:      Dropout probability in the classifier head.
        initial_lr:        Peak learning rate for the cosine decay schedule.
        num_epochs_phase1: Used to calculate total decay steps for the schedule.
        steps_per_epoch:   Training steps per epoch (len(x_train) // batch_size).
        label_smoothing:   Label smoothing epsilon for the loss function.

    Returns:
        model:      Compiled Keras functional model, ready for Phase 1 training.
        base_model: The EfficientNetV2S backbone (needed by unfreeze_top_layers).
    """
    base_model = EfficientNetV2S(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
    )
    base_model.trainable = False  # frozen for Phase 1

    # Build the custom classification head
    inputs = tf.keras.Input(shape=input_shape)

    # Pass training=False so BatchNorm in the base runs in inference mode
    # during Phase 1 — prevents the frozen BN stats from being updated.
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)

    # Cosine decay schedule for Phase 1
    total_steps = num_epochs_phase1 * steps_per_epoch
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=initial_lr,
        decay_steps=total_steps,
        alpha=1e-6,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        metrics=["accuracy"],
    )

    return model, base_model


def unfreeze_top_layers(
    model: tf.keras.Model,
    base_model: tf.keras.Model,
    n_layers: int = 50,
    num_epochs_phase2: int = 15,
    steps_per_epoch: int = 1407,
    finetune_lr: float = 1e-5,
    label_smoothing: float = 0.1,
):
    """
    Unfreeze the top n_layers of the EfficientNetV2S base and recompile
    the model with a lower cosine-decaying learning rate for Phase 2.

    Unfreezing only the top layers (rather than the entire base) keeps the
    low-level feature detectors intact and reduces the risk of catastrophic
    forgetting on the ImageNet weights.

    Args:
        model:             The full Keras model returned by build_model().
        base_model:        The EfficientNetV2S backbone.
        n_layers:          Number of top layers to unfreeze. Default 50.
        num_epochs_phase2: Used to calculate total fine-tune decay steps.
        steps_per_epoch:   Training steps per epoch.
        finetune_lr:       Peak learning rate for fine-tuning. Default 1e-5.
        label_smoothing:   Must match Phase 1 value for consistent training.
    """
    base_model.trainable = True

    # Freeze everything except the last n_layers
    for layer in base_model.layers[:-n_layers]:
        layer.trainable = False

    total_steps = num_epochs_phase2 * steps_per_epoch
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=finetune_lr,
        decay_steps=total_steps,
        alpha=1e-7,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        metrics=["accuracy"],
    )