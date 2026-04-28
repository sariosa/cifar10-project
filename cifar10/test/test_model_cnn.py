# Run with: ./venv/bin/python test/test_model_cnn.py
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.model_CNN import build_baseline_cnn

model = build_baseline_cnn()

# Check the model has the correct number of layers
print("Number of layers:", len(model.layers))

# Check the output shape is 10 (one per CIFAR-10 class)
print("Output shape:", model.output_shape)
assert model.output_shape == (None, 10), "Output shape should be (None, 10)"

# Check the model compiles correctly by printing summary
model.summary()

print("All model tests passed.")
