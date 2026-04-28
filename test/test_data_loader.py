# Run with: ./venv/bin/python test/test_data_loader.py
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.data_loader import load_cifar10

x_train, y_train, x_test, y_test, class_names = load_cifar10()

print("Train shape:", x_train.shape)
print("Test shape:", x_test.shape)
print("Classes:", class_names)
