import os
import sys
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.data_loader import load_cifar10
from src.model_CNN import build_baseline_cnn

print("Starting augmentation training...")

x_train, y_train, x_test, y_test, class_names = load_cifar10()

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    horizontal_flip=True
)

datagen.fit(x_train)

model = build_baseline_cnn()

log_dir = os.path.join(project_root, "logs", "augmentation")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1
)

history = model.fit(
    datagen.flow(x_train, y_train, batch_size=64),
    epochs=15,
    validation_data=(x_test, y_test),
    callbacks=[tensorboard_callback],
    verbose=1
)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)

print("Augmented Test Accuracy:", test_acc)
print("Augmented Test Loss:", test_loss)
print("TensorBoard logs saved to:", log_dir)