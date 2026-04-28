"""
model_improved_cnn.py

What this file does:
--------------------
Defines an improved CNN architecture for CIFAR-10 classification.

How it differs from model_CNN.py (the baseline):
-------------------------------------------------
Baseline (model_CNN.py):
  - Filters: 32 → 64 → 64
  - Activations: ReLU (inline, per Conv2D)
  - No Batch Normalization
  - No Dropout in conv blocks
  - No padding (spatial dims shrink each layer)

This file (model_improved_cnn.py):
  - Residual blocks: 64 → 128 → 256 → 512  (deeper and easier to optimize)
  - Two 3×3 Conv2D layers per residual block  (larger effective receptive field)
  - LeakyReLU(negative_slope=0.1) after BatchNormalization  (stable gradients)
  - GlobalAveragePooling2D instead of Flatten  (far fewer dense parameters)
  - Dense(256) with L2 regularization + Dropout(0.5) in the classifier head
  - Learnable downsampling through stride=2 convolutions in skip blocks

How to use:
    from src.model_improved_cnn import build_improved_cnn
    model = build_improved_cnn()
    model.summary()
"""

from tensorflow.keras import Input, layers, models, regularizers
from tensorflow.keras.layers import LeakyReLU


def residual_block(x, filters, stride=1, kernel_regularizer=None):
    """
    Build a residual block with two 3x3 convolutions and a skip connection.

    Args:
        x: Input tensor
        filters: Number of convolution filters in the block
        stride: Stride for the first convolution. Default 1

    Returns:
        x: Output tensor after residual addition and activation
    """
    shortcut = x

    x = layers.Conv2D(
        filters,
        (3, 3),
        strides=stride,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        kernel_regularizer=kernel_regularizer,
    )(x)
    x = layers.BatchNormalization()(x)
    x = LeakyReLU(negative_slope=0.1)(x)

    x = layers.Conv2D(
        filters,
        (3, 3),
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        kernel_regularizer=kernel_regularizer,
    )(x)
    x = layers.BatchNormalization()(x)

    # Project the shortcut when spatial size or channel count changes
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(
            filters,
            (1, 1),
            strides=stride,
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
            kernel_regularizer=kernel_regularizer,
        )(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = LeakyReLU(negative_slope=0.1)(x)

    return x


def build_improved_cnn(
    input_shape=(32, 32, 3),
    num_classes=10,
    dropout_rate=0.5,
    conv_l2_weight=5e-5,
    dense_l2_weight=1e-4,
):
    """
    Build the improved CNN model.

    Args:
        input_shape: Shape of each input image Default 32, 32, 3
        num_classes: Number of output classes. Default 10
        dropout_rate: Dropout rate used before the output layer. Default 0.5
        conv_l2_weight: L2 regularization for Conv2D kernels. Default 5e-5
        dense_l2_weight: L2 regularization for the Dense layer. Default 1e-4

    Returns:
        model: Uncompiled Keras functional model
    """
    conv_regularizer = (
        regularizers.l2(conv_l2_weight) if conv_l2_weight > 0 else None
    )
    dense_regularizer = (
        regularizers.l2(dense_l2_weight) if dense_l2_weight > 0 else None
    )

    inputs = Input(shape=input_shape)

    # Define the input stem for CIFAR-10 images
    # The bias term is omitted because BatchNormalization follows immediately.
    x = layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        kernel_regularizer=conv_regularizer,
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = LeakyReLU(negative_slope=0.1)(x)

    # First residual block keeps the original 32x32 spatial size
    x = residual_block(x, 64, stride=1, kernel_regularizer=conv_regularizer)

    # Deeper residual stack with learnable downsampling
    x = residual_block(x, 128, stride=2, kernel_regularizer=conv_regularizer)
    x = residual_block(x, 256, stride=2, kernel_regularizer=conv_regularizer)
    x = residual_block(x, 512, stride=2, kernel_regularizer=conv_regularizer)

    # Collapse each feature map to one value before the dense classifier
    x = layers.GlobalAveragePooling2D()(x)

    # Dense feature layer before classification
    x = layers.Dense(
        256,
        use_bias=False,
        kernel_regularizer=dense_regularizer,
        kernel_initializer="he_normal",
    )(x)
    x = layers.BatchNormalization()(x)
    x = LeakyReLU(negative_slope=0.1)(x)

    # Drop units during training to reduce overfitting
    x = layers.Dropout(dropout_rate)(x)

    # Output class probabilities
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="improved_cnn")

    return model


if __name__ == "__main__":
    # Build the model and print a quick architecture summary
    model = build_improved_cnn()
    model.summary()
    print("\nModel built successfully.")
    print(f"Total parameters: {model.count_params():,}")
    print(f"Output shape:     {model.output_shape}")
