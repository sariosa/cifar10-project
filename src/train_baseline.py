"""
train_baseline.py (updated)

Changes from original:
  - Saves test accuracy and loss to outputs/baseline_results.json after training.
    This lets train_improved.py read the real baseline number instead of using
    the hardcoded 0.70 placeholder.
  - Everything else (training loop, model, epochs) is identical to the original.

This file does three things:
1. Loads and preprocesses the CIFAR-10 dataset.
2. Trains the baseline CNN model on the training data.
3. Evaluates the trained model on the test set and saves it
   to the outputs folder.

Why this file exists:
- It keeps the full baseline CNN workflow in one place.
- It combines model training, evaluation, and saving.
- It makes the baseline experiment easier to run and manage.

How to run:
    python scripts/train_baseline.py

What you will see:
- Training progress for each epoch
- Final test accuracy printed in the terminal
- A saved model file in:
    outputs/CNN.keras

How to inspect the saved model later:
    python

Then inside Python:
    from tensorflow.keras.models import load_model
    model = load_model("outputs/CNN.keras")
    model.summary()
"""


import os
import sys
import matplotlib.pyplot as plt
import json

# Set the project root so imports from the src folder work correctly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.data_loader import load_cifar10
from src.model_CNN import build_baseline_cnn


def train_baseline_model():
    """
    Load CIFAR-10, train the baseline CNN, evaluate it,
    and save the trained model.

    Returns:
        model: Trained CNN model
        history: Training history returned by model.fit()
    """
    print("Starting baseline CNN training...")

    # Load CIFAR-10 dataset
    x_train, y_train, x_test, y_test, class_names = load_cifar10()

    # Build the baseline CNN model
    model = build_baseline_cnn()

    # Train the model
    history = model.fit(
        x_train,
        y_train,
        epochs=10,
        batch_size=64,
        validation_split=0.2,
        verbose=1
    )

    # Evaluate the model on the test set
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)

    print("Baseline Test Accuracy:", test_acc)
    print("Baseline Test Loss:", test_loss)

    # Create outputs folder if it does not exist
    output_dir = os.path.join(project_root, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Save the trained model
    save_path = os.path.join(output_dir, "CNN.keras")
    model.save(save_path)

    print(f"Saved model to: {save_path}")

    # ── NEW: persist results so train_improved.py can read the real baseline ──
    results = {
        "test_accuracy": round(float(test_acc), 4),
        "test_loss":     round(float(test_loss), 4),
        "epochs":        10,
        "batch_size":    64,
    }
    results_path = os.path.join(output_dir, "baseline_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")

    # Plot and save training curves
    plot_training_curves(history, output_dir)

    return model, history


def plot_training_curves(history, output_dir):
    """
    Plot and save accuracy and loss curves from baseline CNN training.

    Shows:
    - Training vs validation accuracy over epochs
    - Training vs validation loss over epochs

    Args:
        history: Training history returned by model.fit()
        output_dir: Folder to save the plot
    """
    epochs = range(1, len(history.history["accuracy"]) + 1)

    plt.figure(figsize=(12, 4))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.history["accuracy"], label="Train Accuracy")
    plt.plot(epochs, history.history["val_accuracy"], label="Val Accuracy")
    plt.title("Baseline CNN — Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history["loss"], label="Train Loss")
    plt.plot(epochs, history.history["val_loss"], label="Val Loss")
    plt.title("Baseline CNN — Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    save_path = os.path.join(output_dir, "baseline_training_curves.png")
    plt.savefig(save_path)
    plt.show()
    print(f"Training curves saved to: {save_path}")


def main():
    """
    Main function:
    - loads the dataset
    - trains the baseline CNN
    - evaluates the model
    - saves the trained model
    """
    train_baseline_model()


# Run the script only when executed directly
if __name__ == "__main__":
    main()