"""
train_improved.py

What this file does:
--------------------
Trains the improved CNN (model_improved_cnn.py) on CIFAR-10 and compares
its final test accuracy against the baseline CNN and the earlier improved run.

How it differs from train_baseline.py:
---------------------------------------
train_baseline.py:
  - Trains for a fixed 10 epochs, no callbacks
  - Fixed Adam learning rate (default 1e-3 throughout)
  - validation_split=0.2 (no augmentation during training)

This file (train_improved.py):
  - Uses the residual improved CNN from model_improved_cnn.py
  - Uses augmented data from augmentation.py (same pipeline)
  - Creates a validation split from the training set
  - Uses cosine decay learning rate scheduling instead of ReduceLROnPlateau
  - Uses label smoothing (0.1) to reduce overconfident predictions
  - Uses a larger batch size (128) and longer patience for the deeper model
  - Keeps the test set untouched until final evaluation
  - Loads saved baseline and earlier improved results and prints a comparison table

Why these changes help the residual model:
  - Cosine decay reduces the learning rate smoothly across training rather than
    waiting for a plateau trigger, which usually fits deeper residual models better.
  - Label smoothing regularises the classifier head by discouraging overconfident
    softmax outputs, which can improve calibration and generalisation.

How to run:
    python src/train_improved.py

What you will see:
  - Per-epoch training progress (loss + accuracy)
  - Final comparison table: baseline vs earlier improved vs current improved
  - Saved model: outputs/CNN_improved_v3.keras
  - Training curves: outputs/improved_v3_training_curves.png
"""

import os
import sys
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Set the project root so imports from the src folder work correctly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.data_loader import load_cifar10
from src.augmentation import create_datagen
from src.model_improved_cnn import build_improved_cnn

# Centralise all hyperparameters here so they are easy to find, change, and log.
CONFIG = {
    "epochs":              80,      # longer schedule so cosine decay has time to work
    "batch_size":          128,     # larger batches give steadier gradients for the deeper model
    "learning_rate":       1e-3,    # initial Adam learning rate before cosine decay
    "dropout_rate":        0.5,     # stronger regularisation in the classifier head
    "conv_l2_weight":      5e-5,    # L2 penalty on Conv2D kernels in the trunk
    "dense_l2_weight":     1e-4,    # L2 penalty on the Dense layer in model_improved_cnn.py
    "label_smoothing":     0.1,     # reduce overconfidence in the softmax predictions
    "validation_split":    0.1,     # fraction of the original training set reserved for validation
    "random_seed":         42,      # seed for reproducible train / validation splitting
    "early_stop_patience": 12,      # deeper residual models usually need more time to settle
    "cutout_length":       8,       # side length of the cutout square
    "cutout_n_holes":      1,       # number of cutout squares per image
}

# ── Data split ────────────────────────────────────────────────────────────────

def split_train_validation(x_train, y_train, validation_split, seed):
    """
    Split the original training set into train and validation subsets.

    The split is stratified by class so the label distribution stays similar
    in both subsets.

    Args:
        x_train: Full CIFAR-10 training images
        y_train: One-hot encoded CIFAR-10 training labels
        validation_split: Fraction reserved for validation
        seed: Random seed for reproducibility

    Returns:
        x_train_split, y_train_split, x_val, y_val
    """
    rng = np.random.default_rng(seed)
    class_ids = np.argmax(y_train, axis=1)

    train_indices = []
    val_indices = []

    for class_id in range(y_train.shape[1]):
        class_indices = np.where(class_ids == class_id)[0]
        rng.shuffle(class_indices)

        val_count = int(len(class_indices) * validation_split)
        val_indices.extend(class_indices[:val_count])
        train_indices.extend(class_indices[val_count:])

    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    return (
        x_train[train_indices],
        y_train[train_indices],
        x_train[val_indices],
        y_train[val_indices],
    )

# ── Callbacks ─────────────────────────────────────────────────────────────────

def build_callbacks(output_dir):
    """
    Build the training callbacks.

    EarlyStopping:
        Monitors val_accuracy. If it does not improve for patience epochs,
        training stops and the best weights are restored.
        This prevents wasted compute and overfitting.

    ModelCheckpoint:
        Saves the model only when val_accuracy improves. This means the saved
        file always contains the best weights seen during training, not just
        the final weights.

    TensorBoard:
        Logs metrics for visualisation. After training, run:
            tensorboard --logdir=logs/improved_v3

    Args:
        output_dir: Directory to save the best model checkpoint.

    Returns:
        Tuple of:
        - List of Keras callback objects
        - Path to the best-model checkpoint
    """
    # Create log directory for TensorBoard
    log_dir = os.path.join(project_root, "logs", "improved_v3")
    os.makedirs(log_dir, exist_ok=True)

    checkpoint_path = os.path.join(output_dir, "CNN_improved_v3.keras")

    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=CONFIG["early_stop_patience"],
        restore_best_weights=True,  # revert to best checkpoint on stop
        verbose=1,
    )
    # ModelCheckpoint monitors val_accuracy and saves the best model to disk
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    )
    # TensorBoard callback for logging metrics
    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=0,
    )

    return [early_stopping, model_checkpoint, tensorboard], checkpoint_path

# ── Training ──────────────────────────────────────────────────────────────────

def train_improved_model():
    """
    Full training pipeline for the improved CNN.

    Steps:
      1. Load CIFAR-10 and create train / validation / test splits
      2. Apply augmentation to the training split only
      3. Build improved CNN with CONFIG hyperparameters
      4. Train with cosine decay LR scheduling and callbacks
      5. Evaluate on test set
      6. Save model, curves, and CONFIG to outputs/
      7. Print accuracy comparison vs baseline and earlier improved run

    Returns:
        model:    Trained Keras model.
        history:  Training History object.
        test_acc: Final test accuracy (float).
    """
    print("=" * 60)
    print("  Training improved CNN v3 on CIFAR-10")
    print("=" * 60)
    print("\nHyperparameters:")
    for k, v in CONFIG.items():
        print(f"  {k:<24} {v}")
    print()

    # Seed Python, NumPy, and TensorFlow for more reproducible runs.
    random.seed(CONFIG["random_seed"])
    np.random.seed(CONFIG["random_seed"])
    tf.keras.utils.set_random_seed(CONFIG["random_seed"])

    # ── 1. Data ───────────────────────────────────────────────────────────────
    x_train_full, y_train_full, x_test, y_test, _ = load_cifar10()

    x_train, y_train, x_val, y_val = split_train_validation(
        x_train_full,
        y_train_full,
        validation_split=CONFIG["validation_split"],
        seed=CONFIG["random_seed"],
    )

    datagen = create_datagen(
        cutout_length=CONFIG["cutout_length"],
        cutout_n_holes=CONFIG["cutout_n_holes"],
    )

    # ── 2. Model ──────────────────────────────────────────────────────────────
    model = build_improved_cnn(
        dropout_rate=CONFIG["dropout_rate"],
        conv_l2_weight=CONFIG["conv_l2_weight"],
        dense_l2_weight=CONFIG["dense_l2_weight"],
    )

    # Cosine decay lowers the learning rate smoothly across all update steps.
    steps_per_epoch = max(1, int(np.ceil(len(x_train) / CONFIG["batch_size"])))
    total_steps = CONFIG["epochs"] * steps_per_epoch
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=CONFIG["learning_rate"],
        decay_steps=total_steps,
        alpha=1e-6,
    )

    # Override the compile so CONFIG remains the single source of truth.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=CONFIG["label_smoothing"],
        ),
        metrics=["accuracy"],
    )

    model.summary()

    # ── 3. Callbacks ──────────────────────────────────────────────────────────
    output_dir = os.path.join(project_root, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    callbacks, checkpoint_path = build_callbacks(output_dir)

    # ── 4. Train ──────────────────────────────────────────────────────────────
    print("\nStarting training...")
    history = model.fit(
        datagen.flow(
            x_train,
            y_train,
            batch_size=CONFIG["batch_size"],
            shuffle=True,
            seed=CONFIG["random_seed"],
        ),
        epochs=CONFIG["epochs"],
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        verbose=1,
    )

    # ── 5. Evaluate ───────────────────────────────────────────────────────────
    # Restore the best checkpoint explicitly before final evaluation and save.
    model.load_weights(checkpoint_path)
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
    print(f"\nImproved CNN v3 — Test Accuracy : {test_acc:.4f}")
    print(f"Improved CNN v3 — Test Loss     : {test_loss:.4f}")

    # ── 6. Save artefacts ─────────────────────────────────────────────────────
    # Model (best checkpoint already saved by ModelCheckpoint callback above;
    # this saves the final model state as well for reference)
    final_model_path = os.path.join(output_dir, "CNN_improved_v3_final.keras")
    model.save(final_model_path)
    print(f"Final model saved to: {final_model_path}")

    # Save the CONFIG and final accuracy as JSON for the results log
    results = {
        **CONFIG,
        "test_accuracy": round(float(test_acc), 4),
        "test_loss": round(float(test_loss), 4),
    }
    results_path = os.path.join(output_dir, "improved_v3_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to:   {results_path}")

    # Training curves
    plot_training_curves(history, output_dir)

    # Accuracy comparison
    print_comparison(test_acc)

    return model, history, test_acc

# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_training_curves(history, output_dir):
    """
    Plot and save accuracy and loss curves for the improved CNN.

    Args:
        history:    Training History object from model.fit().
        output_dir: Directory to save the plot.
    """
    epochs = range(1, len(history.history["accuracy"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(epochs, history.history["accuracy"],     label="Train")
    axes[0].plot(epochs, history.history["val_accuracy"], label="Validation")
    axes[0].set_title("Improved CNN v3 — Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    # Loss
    axes[1].plot(epochs, history.history["loss"],     label="Train")
    axes[1].plot(epochs, history.history["val_loss"], label="Validation")
    axes[1].set_title("Improved CNN v3 — Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()
    save_path = os.path.join(output_dir, "improved_v3_training_curves.png")
    plt.savefig(save_path)
    plt.show()
    print(f"Training curves saved to: {save_path}")

# ── Comparison ────────────────────────────────────────────────────────────────

def print_comparison(acc_improved):
    """
    Print a formatted accuracy comparison table.

    Loads the baseline accuracy from outputs/baseline_results.json if available,
    loads the earlier improved accuracy from outputs/improved_results.json if
    available, and prints the current run as the new improved model.

    Args:
        acc_improved: Test accuracy of the improved model (float).
    """
    output_dir = os.path.join(project_root, "outputs")

    # Try to read a previously saved baseline result
    baseline_path = os.path.join(output_dir, "baseline_results.json")
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            acc_baseline = json.load(f).get("test_accuracy", 0.70)
    else:
        # Fall back to the value hardcoded in transfer_learning.py
        acc_baseline = 0.70

    # Try to read a previous improved result for the v1 comparison
    previous_improved = None
    previous_path = os.path.join(output_dir, "improved_results.json")
    if os.path.exists(previous_path):
        with open(previous_path) as f:
            previous_improved = json.load(f).get("test_accuracy")

    diff = acc_improved - acc_baseline

    print("\n" + "=" * 55)
    print("  ACCURACY COMPARISON")
    print("=" * 55)
    print(f"  Baseline CNN        : {acc_baseline:.4f}  ({acc_baseline*100:.1f}%)")
    if previous_improved is not None:
        print(f"  Improved CNN v1     : {previous_improved:.4f}  ({previous_improved*100:.1f}%)")
    print(f"  Improved CNN v3     : {acc_improved:.4f}  ({acc_improved*100:.1f}%)")
    print(f"  Gain vs baseline    : {diff:+.4f}  ({diff*100:+.1f}%)")
    print("=" * 55)

# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    train_improved_model()

if __name__ == "__main__":
    main()
