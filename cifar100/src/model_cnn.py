"""
model_cnn.py

CNN architecture for CIFAR-100 classification.

Lessons applied from CIFAR-10:
--------------------------------
1. Skip the weak baseline — start with the improved architecture directly.
2. Higher capacity: 4 conv blocks (128→256→512→512) vs 3 in CIFAR-10's
   improved model. CIFAR-100 needs more representational power for 100 classes.
3. Higher dropout (0.5 default): only 500 images per class vs 5,000 in CIFAR-10,
   so overfitting risk is much greater.
4. L2 weight regularization on Dense layers for extra overfitting control.
5. BatchNormalization + LeakyReLU kept — they worked well on CIFAR-10.
6. GlobalAveragePooling instead of Flatten: fewer parameters, less overfitting
   compared to CIFAR-10's Flatten approach.

How to use:
    from cifar100.src.model_cnn import build_cnn
    model = build_cnn()
    model.summary()
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.layers import LeakyReLU


def build_cnn(input_shape=(32, 32, 3), num_classes=100, dropout_rate=0.5):
    """
    Build and compile the CNN for CIFAR-100.

    Architecture:
    - 4 conv blocks with BatchNorm + LeakyReLU (128 → 256 → 512 → 512 filters)
    - Two Conv2D layers per block for deeper feature extraction
    - MaxPooling after blocks 1, 2, and 3
    - GlobalAveragePooling to reduce parameters before classifier
    - Dense(512) + Dropout before the 100-class softmax output

    Args:
        input_shape:  Image shape. Default (32, 32, 3).
        num_classes:  Number of output classes. Default 100.
        dropout_rate: Dropout probability before output layer. Default 0.5.

    Returns:
        Compiled Keras Sequential model.
    """
    l2 = regularizers.l2(1e-4)

    model = models.Sequential([
        layers.Input(shape=input_shape),

        # Block 1 — 128 filters
        layers.Conv2D(128, (3, 3), padding="same", use_bias=False),
        layers.BatchNormalization(),
        LeakyReLU(negative_slope=0.1),
        layers.Conv2D(128, (3, 3), padding="same", use_bias=False),
        layers.BatchNormalization(),
        LeakyReLU(negative_slope=0.1),
        layers.MaxPooling2D((2, 2)),

        # Block 2 — 256 filters
        layers.Conv2D(256, (3, 3), padding="same", use_bias=False),
        layers.BatchNormalization(),
        LeakyReLU(negative_slope=0.1),
        layers.Conv2D(256, (3, 3), padding="same", use_bias=False),
        layers.BatchNormalization(),
        LeakyReLU(negative_slope=0.1),
        layers.MaxPooling2D((2, 2)),

        # Block 3 — 512 filters
        layers.Conv2D(512, (3, 3), padding="same", use_bias=False),
        layers.BatchNormalization(),
        LeakyReLU(negative_slope=0.1),
        layers.Conv2D(512, (3, 3), padding="same", use_bias=False),
        layers.BatchNormalization(),
        LeakyReLU(negative_slope=0.1),
        layers.MaxPooling2D((2, 2)),

        # Block 4 — 512 filters (no pooling — spatial dims already small at 4x4)
        layers.Conv2D(512, (3, 3), padding="same", use_bias=False),
        layers.BatchNormalization(),
        LeakyReLU(negative_slope=0.1),

        # GlobalAveragePooling: averages each feature map to one value
        # Fewer parameters than Flatten, helps generalization
        layers.GlobalAveragePooling2D(),

        # Dense classifier with L2 regularization
        layers.Dense(512, kernel_regularizer=l2, use_bias=False),
        layers.BatchNormalization(),
        LeakyReLU(negative_slope=0.1),
        layers.Dropout(dropout_rate),

        # 100-class output
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":
    model = build_cnn()
    model.summary()
    print(f"\nTotal parameters: {model.count_params():,}")
