# Run with: ./venv/bin/python test/test_transfer_learning.py
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.transfer_learning import build_transfer_model, resize_images
from src.data_loader import load_cifar10
import numpy as np

# Check the transfer learning model builds correctly
model, base_model = build_transfer_model()
print("Model built successfully.")
print("Output shape:", model.output_shape)
assert model.output_shape == (None, 10), "Output shape should be (None, 10)"

# Check the base model is frozen
assert base_model.trainable == False, "Base model should be frozen initially"
print("Base model is frozen:", not base_model.trainable)

# Check image resizing works correctly
sample_images = np.random.rand(4, 32, 32, 3).astype("float32")
resized = resize_images(sample_images, target_size=(224, 224))
print("Resized image shape:", resized.shape)
assert resized.shape == (4, 224, 224, 3), "Resized shape should be (4, 224, 224, 3)"

print("All transfer learning tests passed.")
