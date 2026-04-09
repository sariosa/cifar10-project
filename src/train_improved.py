"""
train_improved.py

What this file does:
--------------------
Trains the improved CNN (model_improved_cnn.py) on CIFAR-10 and compares
its final test accuracy against the baseline CNN.

How it differs from train_baseline.py:
---------------------------------------
train_baseline.py:
  - Trains for a fixed 10 epochs, no callbacks
  - Fixed Adam learning rate (default 1e-3 throughout)
  - validation_split=0.2 (no augmentation during training)

This file (train_improved.py):
  - Uses augmented data from augmentation.py (same pipeline)
  - Creates a validation split from the training set
  - ReduceLROnPlateau: halves the LR when val_accuracy stalls for 3 epochs
  - EarlyStopping: stops training when val_accuracy stops improving
  - ModelCheckpoint: saves the single best model based on val_accuracy
  - Runs up to 50 epochs but will stop early if the model converges
  - Keeps the test set untouched until final evaluation
  - Loads the saved baseline accuracy and prints a clear comparison table

Callback rationale (grounded in the SGD lecture):
  - ReduceLROnPlateau mimics the effect of a decaying learning rate schedule.
    The SGD lecture explains that a fixed LR can overshoot the minimum once
    training is close to convergence — reducing it lets the optimizer settle.
  - EarlyStopping prevents the overfitting described in the capacity lecture:
    "a model with too much capacity will eventually memorise training data."

How to run:
    python src/train_improved.py

What you will see:
  - Per-epoch training progress (loss + accuracy)
  - Callbacks firing (LR reductions, early stop trigger)
  - Final comparison table: baseline vs improved test accuracy
  - Saved model: outputs/CNN_improved.keras
  - Training curves: outputs/improved_training_curves.png
"""

import os
import sys
import json
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
    "epochs":           50,      # max epochs — EarlyStopping will likely end sooner
    "batch_size":       64,      # Batch size for training. Larger batches can speed up training but may require more memory. 64 is a common default for CIFAR-10.
    "learning_rate":    1e-3,    # Adam initial LR — will be reduced by ReduceLROnPlateau if val_accuracy plateaus
    "dropout_rate":     0.4,     # Dropout probability before the final Dense layer. Increase toward 0.5 if you observe overfitting in training curves.
    "validation_split": 0.1,     # fraction of the original training set reserved for validation
    "random_seed":      42,      # seed for reproducible train / validation splitting
    "lr_reduce_factor": 0.5,     # multiply LR by this when plateau detected
    "lr_reduce_patience": 3,     # epochs to wait before reducing LR
    "lr_min":           1e-6,    # LR floor — never reduce below this
    "early_stop_patience": 7,    # epochs to wait before stopping training
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

    ReduceLROnPlateau:
        Monitors val_accuracy. If it does not improve for patience epochs,
        the learning rate is multiplied by factor. This helps the model
        converge without manually scheduling the LR.

    EarlyStopping:
        Monitors val_accuracy. If it does not improve for patience epochs,
        training stops and the best weights are restored.
        This prevents wasted compute and overfitting.

    ModelCheckpoint:
        Saves the model only when val_accuracy improves. This means the saved
        file always contains the best weights seen during training, not just
        the final weights (which may be slightly worse if the model overfit
        in the last few epochs before early stopping triggered).

    TensorBoard:
        Logs metrics for visualisation. After training, run:
            tensorboard --logdir=logs/improved

    Args:
        output_dir: Directory to save the best model checkpoint.

    Returns:
        List of Keras callback objects.
    """
    # Create log directory for TensorBoard
    log_dir = os.path.join(project_root, "logs", "improved")
    os.makedirs(log_dir, exist_ok=True)

    checkpoint_path = os.path.join(output_dir, "CNN_improved.keras")

    # Callbacks
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_accuracy",
        factor=CONFIG["lr_reduce_factor"],
        patience=CONFIG["lr_reduce_patience"],
        min_lr=CONFIG["lr_min"],
        verbose=1,
    )
    # EarlyStopping monitors val_accuracy
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
        histogram_freq=1,
    )

    return [reduce_lr, early_stopping, model_checkpoint, tensorboard]

# ── Training ──────────────────────────────────────────────────────────────────

def train_improved_model():
    """
    Full training pipeline for the improved CNN.

    Steps:
      1. Load CIFAR-10 and create train / validation / test splits
      2. Apply augmentation to the training split only
      3. Build improved CNN with CONFIG hyperparameters
      4. Train with callbacks (ReduceLROnPlateau + EarlyStopping + Checkpoint)
      5. Evaluate on test set
      6. Save model, curves, and CONFIG to outputs/
      7. Print accuracy comparison vs baseline

    Returns:
        model:    Trained Keras model.
        history:  Training History object.
        test_acc: Final test accuracy (float).
    """
    print("=" * 60)
    print("  Training improved CNN on CIFAR-10")
    print("=" * 60)
    print("\nHyperparameters:")
    for k, v in CONFIG.items():
        print(f"  {k:<24} {v}")
    print()

    # ── 1. Data ───────────────────────────────────────────────────────────────
    x_train_full, y_train_full, x_test, y_test, _ = load_cifar10()

    x_train, y_train, x_val, y_val = split_train_validation(
        x_train_full,
        y_train_full,
        validation_split=CONFIG["validation_split"],
        seed=CONFIG["random_seed"],
    )

    datagen = create_datagen()
    datagen.fit(x_train)

    # ── 2. Model ──────────────────────────────────────────────────────────────
    model = build_improved_cnn(dropout_rate=CONFIG["dropout_rate"])

    # Override the compile with the configured learning rate.
    # build_improved_cnn() compiles with adam default; we recompile here so
    # CONFIG["learning_rate"] is the single source of truth.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=CONFIG["learning_rate"]),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    # ── 3. Callbacks ──────────────────────────────────────────────────────────
    output_dir = os.path.join(project_root, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    callbacks = build_callbacks(output_dir)

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
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
    print(f"\nImproved CNN — Test Accuracy : {test_acc:.4f}")
    print(f"Improved CNN — Test Loss     : {test_loss:.4f}")

    # ── 6. Save artefacts ─────────────────────────────────────────────────────
    # Model (best checkpoint already saved by ModelCheckpoint callback above;
    # this saves the final model state as well for reference)
    final_model_path = os.path.join(output_dir, "CNN_improved_final.keras")
    model.save(final_model_path)
    print(f"Final model saved to: {final_model_path}")

    # Save the CONFIG and final accuracy as JSON for the results log
    results = {**CONFIG, "test_accuracy": round(float(test_acc), 4), "test_loss": round(float(test_loss), 4)}
    results_path = os.path.join(output_dir, "improved_results.json")
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
    axes[0].set_title("Improved CNN — Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    # Loss
    axes[1].plot(epochs, history.history["loss"],     label="Train")
    axes[1].plot(epochs, history.history["val_loss"], label="Validation")
    axes[1].set_title("Improved CNN — Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()
    save_path = os.path.join(output_dir, "improved_training_curves.png")
    plt.savefig(save_path)
    plt.show()
    print(f"Training curves saved to: {save_path}")

# ── Comparison ────────────────────────────────────────────────────────────────

def print_comparison(acc_improved):
    """
    Print a formatted accuracy comparison table.

    Loads the baseline accuracy from outputs/improved_results.json if available,
    otherwise falls back to the hardcoded reference value (0.70) from
    transfer_learning.py.

    Args:
        acc_improved: Test accuracy of the improved model (float).
    """
    # Try to read a previously saved baseline result
    baseline_path = os.path.join(project_root, "outputs", "baseline_results.json")
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            acc_baseline = json.load(f).get("test_accuracy", 0.70)
    else:
        # Fall back to the value hardcoded in transfer_learning.py
        acc_baseline = 0.70

    diff = acc_improved - acc_baseline
    direction = "improvement" if diff >= 0 else "regression"

    print("\n" + "=" * 50)
    print("  ACCURACY COMPARISON")
    print("=" * 50)
    print(f"  Baseline CNN   : {acc_baseline:.4f}  ({acc_baseline*100:.1f}%)")
    print(f"  Improved CNN   : {acc_improved:.4f}  ({acc_improved*100:.1f}%)")
    print(f"  Difference     : {diff:+.4f}  ({diff*100:+.1f}%)  ← {direction}")
    print("=" * 50)

# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    train_improved_model()

if __name__ == "__main__":
    main()
