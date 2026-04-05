import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.augmentation import create_datagen
from src.data_loader import load_cifar10

# Check the augmentation generator is created correctly
datagen = create_datagen()
print("Augmentation generator created:", datagen)

# Check augmentation settings
print("Rotation range:", datagen.rotation_range)
print("Zoom range:", datagen.zoom_range)
print("Horizontal flip:", datagen.horizontal_flip)

assert datagen.rotation_range == 10, "Rotation range should be 10"
assert datagen.horizontal_flip == True, "Horizontal flip should be enabled"
assert datagen.zoom_range == [0.9, 1.1], "Zoom range should be 0.1"

# Check it can generate a batch without errors
x_train, y_train, _, _, _ = load_cifar10()
datagen.fit(x_train[:100])
batch = next(datagen.flow(x_train[:32], y_train[:32], batch_size=8))
print("Augmented batch shape:", batch[0].shape)
assert batch[0].shape == (8, 32, 32, 3), "Batch image shape should be (8, 32, 32, 3)"

print("All augmentation tests passed.")
