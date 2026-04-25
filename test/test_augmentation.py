# Run with: ./venv/bin/python test/test_augmentation.py
import sys
import os
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.augmentation import create_datagen
from src.data_loader import load_cifar10

# Check the augmentation generator is created correctly
datagen = create_datagen()
print("Augmentation generator created:", datagen)

# Check augmentation settings
print("Rotation range:", datagen.rotation_range)
print("Width shift range:", datagen.width_shift_range)
print("Height shift range:", datagen.height_shift_range)
print("Zoom range:", datagen.zoom_range)
print("Horizontal flip:", datagen.horizontal_flip)
print("Has preprocessing function:", datagen.preprocessing_function is not None)

assert datagen.rotation_range == 15, "Rotation range should be 15"
assert datagen.width_shift_range == 0.125, "Width shift range should be 0.125"
assert datagen.height_shift_range == 0.125, "Height shift range should be 0.125"
assert datagen.horizontal_flip is True, "Horizontal flip should be enabled"
assert np.allclose(datagen.zoom_range, [0.85, 1.15]), "Zoom range should be [0.85, 1.15]"
assert datagen.preprocessing_function is not None, "Cutout preprocessing should be enabled"

# Check it can generate a batch without errors
x_train, y_train, _, _, _ = load_cifar10()
datagen.fit(x_train[:100])
batch = next(datagen.flow(x_train[:32], y_train[:32], batch_size=8))
print("Augmented batch shape:", batch[0].shape)
assert batch[0].shape == (8, 32, 32, 3), "Batch image shape should be (8, 32, 32, 3)"

print("All augmentation tests passed.")
