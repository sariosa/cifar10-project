"""
augmentation.py

Data augmentation pipeline for CIFAR-100.

Why stronger augmentation than CIFAR-10:
-----------------------------------------
CIFAR-100 has only 500 images per class (vs 5,000 in CIFAR-10).
With so few examples per class, the model will memorise training data quickly
unless we artificially expand the variety. Stronger augmentation acts as a
regularizer and helps the model generalise to unseen images.

Changes vs CIFAR-10 augmentation:
- rotation_range:     15  (was 10)
- width_shift_range:  0.1 (was 0.05)
- height_shift_range: 0.1 (was 0.05)
- zoom_range:         0.15 (was 0.1)
- horizontal_flip:    True (same)
- Added shear_range:  0.1 (new — adds perspective distortion)

How to use:
    from cifar100.src.augmentation import create_datagen
    datagen = create_datagen()
    datagen.fit(x_train)
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator


def create_datagen():
    """
    Create and return the augmentation generator for CIFAR-100.

    Returns:
        ImageDataGenerator configured with stronger augmentation than CIFAR-10.
    """
    return ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
    )
