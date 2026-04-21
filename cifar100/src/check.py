"""
Quick step-by-step check to find where the hang is.
Run: python cifar100/src/check.py
"""
import sys

print("Step 1: importing numpy...", flush=True)
import numpy as np
print("OK", flush=True)

print("Step 2: importing tensorflow...", flush=True)
import tensorflow as tf
print(f"OK — TF version: {tf.__version__}", flush=True)

print("Step 3: loading CIFAR-100 data...", flush=True)
from tensorflow.keras.datasets import cifar100
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
print(f"OK — train shape: {x_train.shape}", flush=True)

print("Step 4: loading EfficientNetB3 weights...", flush=True)
from tensorflow.keras.applications import EfficientNetB3
base = EfficientNetB3(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
print(f"OK — base loaded, params: {base.count_params():,}", flush=True)

print("\nAll checks passed. Ready to train.", flush=True)
