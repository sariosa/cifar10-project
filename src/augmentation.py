"""
augmentation.py

This file does two things:
1. Visualizes CIFAR-10 image augmentation by showing original images
   alongside augmented versions.
2. Trains the baseline CNN model using augmented training data and
   saves TensorBoard logs for monitoring performance.

Why this file exists:
- It keeps augmentation preview and augmentation training together
  in one place.
- The augmentation settings are defined once and reused for both
  visualization and training.
  
  after training completed use the command below in terminal 
  to see the log results in tensorboard local host:
  tensorboard --logdir=logs/augmentation 
  
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set the project root so imports from the src folder work correctly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.data_loader import load_cifar10
from src.model_CNN import build_baseline_cnn


def make_cutout_fn(length=8, n_holes=1):
    """
    Build a per-image cutout function for ImageDataGenerator.

    Args:
        length: Side length of each cutout square.
        n_holes: Number of squares to mask per image.
    """
    def cutout_fn(image):
        if length <= 0 or n_holes <= 0:
            return image

        h, w = image.shape[:2]
        mask = np.ones_like(image, dtype=np.float32)

        for _ in range(n_holes):
            cy = np.random.randint(0, h)
            cx = np.random.randint(0, w)

            y1 = max(0, cy - length // 2)
            y2 = min(h, cy + length // 2)
            x1 = max(0, cx - length // 2)
            x2 = min(w, cx + length // 2)

            mask[y1:y2, x1:x2, :] = 0.0

        return image * mask

    return cutout_fn


def create_datagen(cutout_length=8, cutout_n_holes=1):
    """
    Create and return the image augmentation generator.

    The same augmentation settings are used for both:
    - previewing augmented images
    - training the CNN model
    """
    return ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.125,
        height_shift_range=0.125,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
        preprocessing_function=make_cutout_fn(cutout_length, cutout_n_holes),
    )


def preview_augmentation(x_train, y_train, class_names, datagen, num_samples=5):
    """
    Display a few original CIFAR-10 images next to their augmented versions.

    Args:
        x_train: Training images
        y_train: One-hot encoded training labels
        class_names: List of CIFAR-10 class names
        datagen: ImageDataGenerator object
        num_samples: Number of samples to display
    """
    # Select a small number of training images for visualization
    sample_images = x_train[:num_samples]

    # Create the plotting window
    plt.figure(figsize=(12, 2 * num_samples))

    for i in range(num_samples):
        # Show the original image
        plt.subplot(num_samples, 2, 2 * i + 1)
        plt.imshow(sample_images[i])
        plt.title(f"Original: {class_names[y_train[i].argmax()]}")
        plt.axis("off")

        # Generate one augmented version of the same image
        aug_iter = datagen.flow(sample_images[i:i+1], batch_size=1, shuffle=False)
        aug_image = next(aug_iter)[0]

        # Show the augmented image
        plt.subplot(num_samples, 2, 2 * i + 2)
        plt.imshow(aug_image)
        plt.title("Augmented")
        plt.axis("off")

    # Adjust spacing and display the plot
    plt.tight_layout()
    plt.show()


def train_with_augmentation(x_train, y_train, x_test, y_test, datagen):
    """
    Train the baseline CNN model using augmented CIFAR-10 training data.

    Also creates TensorBoard logs for tracking training progress.

    Args:
        x_train: Training images
        y_train: Training labels
        x_test: Test images
        y_test: Test labels
        datagen: ImageDataGenerator object

    Returns:
        model: Trained CNN model
        history: Training history returned by model.fit()
    """
    print("Starting augmentation training...")

    # Fit the generator to the training data
    datagen.fit(x_train)

    # Build the baseline CNN model
    model = build_baseline_cnn()

    # Create a folder for TensorBoard logs
    log_dir = os.path.join(project_root, "logs", "augmentation")
    os.makedirs(log_dir, exist_ok=True)

    # TensorBoard callback for logging training details
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1
    )

    # Stop training early if validation accuracy stops improving for 3 epochs
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=3,
        restore_best_weights=True,
        verbose=1
    )

    # Save the best model during training based on validation accuracy
    checkpoint_path = os.path.join(project_root, "outputs", "CNN_augmented_best.keras")
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )

    # Train the model using augmented batches
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=64),
        epochs=15,
        validation_data=(x_test, y_test),
        callbacks=[tensorboard_callback, early_stopping, model_checkpoint],
        verbose=1
    )

    # Evaluate the model on the test set
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)

    print("Augmented Test Accuracy:", test_acc)
    print("Augmented Test Loss:", test_loss)
    print("TensorBoard logs saved to:", log_dir)

    return model, history, test_acc


def train_without_augmentation(x_train, y_train, x_test, y_test):
    """
    Train the baseline CNN model without any augmentation.

    Args:
        x_train: Training images
        y_train: Training labels
        x_test: Test images
        y_test: Test labels

    Returns:
        model: Trained CNN model
        history: Training history returned by model.fit()
        test_acc: Final test accuracy
    """
    print("Starting baseline (no augmentation) training...")

    model = build_baseline_cnn()

    history = model.fit(
        x_train,
        y_train,
        epochs=15,
        batch_size=64,
        validation_data=(x_test, y_test),
        verbose=1
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
    print("Baseline Test Accuracy:", test_acc)
    print("Baseline Test Loss:", test_loss)

    return model, history, test_acc


def plot_comparison(history_baseline, history_augmented, acc_baseline, acc_augmented):
    """
    Plot training curves for baseline vs augmented training and print accuracy comparison.

    Shows:
    - Training accuracy over epochs for both runs
    - Validation accuracy over epochs for both runs

    Args:
        history_baseline: History object from baseline training
        history_augmented: History object from augmented training
        acc_baseline: Final test accuracy of baseline model
        acc_augmented: Final test accuracy of augmented model
    """
    epochs = range(1, len(history_baseline.history["accuracy"]) + 1)

    plt.figure(figsize=(14, 5))

    # --- Training Accuracy ---
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history_baseline.history["accuracy"], label="Baseline Train")
    plt.plot(epochs, history_augmented.history["accuracy"], label="Augmented Train")
    plt.title("Training Accuracy: Baseline vs Augmented")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # --- Validation Accuracy ---
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history_baseline.history["val_accuracy"], label="Baseline Val")
    plt.plot(epochs, history_augmented.history["val_accuracy"], label="Augmented Val")
    plt.title("Validation Accuracy: Baseline vs Augmented")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    os.makedirs(os.path.join(project_root, "outputs"), exist_ok=True)
    plt.savefig(os.path.join(project_root, "outputs", "augmentation_comparison.png"))
    plt.show()

    # Print accuracy comparison
    print("\n--- Accuracy Comparison ---")
    print(f"Baseline  Test Accuracy: {acc_baseline:.4f}")
    print(f"Augmented Test Accuracy: {acc_augmented:.4f}")
    diff = acc_augmented - acc_baseline
    if diff > 0:
        print(f"Augmentation improved accuracy by {diff:.4f}")
    else:
        print(f"Augmentation reduced accuracy by {abs(diff):.4f}")


def main():
    """
    Main function:
    - loads CIFAR-10 data
    - creates the augmentation generator
    - previews augmented images
    - trains baseline CNN (no augmentation) for comparison
    - trains CNN with augmentation
    - plots training curves for both and prints accuracy comparison
    """
    # Load dataset and class labels
    x_train, y_train, x_test, y_test, class_names = load_cifar10()

    # Create augmentation pipeline
    datagen = create_datagen()

    # Show sample augmented images
    preview_augmentation(x_train, y_train, class_names, datagen)

    # Train baseline model (no augmentation)
    _, history_baseline, acc_baseline = train_without_augmentation(
        x_train, y_train, x_test, y_test
    )

    # Train model with augmentation
    _, history_augmented, acc_augmented = train_with_augmentation(
        x_train, y_train, x_test, y_test, datagen
    )

    # Plot and compare both runs
    plot_comparison(history_baseline, history_augmented, acc_baseline, acc_augmented)


# Run the script only when executed directly
if __name__ == "__main__":
    main()
    
