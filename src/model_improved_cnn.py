"""
model_improved_cnn.py

What this file does:
--------------------
Defines and compiles an improved CNN architecture for CIFAR-10 classification.

How it differs from model_CNN.py (the baseline):
-------------------------------------------------
Baseline (model_CNN.py):
  - Filters: 32 → 64 → 64
  - Activations: ReLU (inline, per Conv2D)
  - No Batch Normalization
  - No Dropout in conv blocks
  - No padding (spatial dims shrink each layer)

This file (model_improved_cnn.py):
  - Filters: 64 → 128 → 256  (higher capacity)
  - Activations: LeakyReLU(alpha=0.1)  (no dead neurons)
  - BatchNormalization after every Conv2D  (stable, faster training)
  - Dropout(0.4) in the classifier head before the output layer
  - padding='same' on all Conv2D  (preserves spatial dims through deeper stack)

How to use:
    from src.model_improved_cnn import build_improved_cnn
    model = build_improved_cnn()
    model.summary()
"""

from tensorflow.keras import layers, models
from tensorflow.keras.layers import LeakyReLU

def build_improved_cnn(input_shape=(32, 32, 3), num_classes=10, dropout_rate=0.4):
    """
    Build and compile the improved CNN model.

    Args:
        input_shape: Shape of each input image Default 32, 32, 3
        num_classes: Number of output classes. Default 10
        dropout_rate: Dropout rate used before the output layer. Default 0.4

    Returns:
        model: Compiled Keras Sequential model
    """
    model = models.Sequential([

        # Define the input shape for CIFAR-10 images
        layers.Input(shape=input_shape),

        # First convolution block
        # The bias term is omitted because BatchNormalization follows immediately.
        layers.Conv2D(64, (3, 3), padding="same", use_bias=False),
        layers.BatchNormalization(),
        LeakyReLU(negative_slope=0.1),
        layers.MaxPooling2D((2, 2)),

        # Second convolution block
        layers.Conv2D(128, (3, 3), padding="same", use_bias=False),
        layers.BatchNormalization(),
        LeakyReLU(negative_slope=0.1),
        layers.MaxPooling2D((2, 2)),

        # Third convolution block without another pooling layer
        layers.Conv2D(256, (3, 3), padding="same", use_bias=False),
        layers.BatchNormalization(),
        LeakyReLU(negative_slope=0.1),

        # Flatten feature maps before the dense classifier
        layers.Flatten(),

        # Dense feature layer before classification
        layers.Dense(256, use_bias=False),
        layers.BatchNormalization(),
        LeakyReLU(negative_slope=0.1),

        # Drop units during training to reduce overfitting
        layers.Dropout(dropout_rate),

        # Output class probabilities
        layers.Dense(num_classes, activation="softmax"),
    ])

    # Compile the model for multi-class classification with one-hot labels
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model

if __name__ == "__main__":
    # Build the model and print a quick architecture summary
    model = build_improved_cnn()
    model.summary()
    print("\nModel built successfully.")
    print(f"Total parameters: {model.count_params():,}")
    print(f"Output shape:     {model.output_shape}")
    