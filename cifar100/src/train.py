"""
train.py

Training pipeline for the CIFAR-100 CNN.

Lessons applied from CIFAR-10:
--------------------------------
- No weak baseline run first — augmentation and callbacks are on from epoch 1.
- ReduceLROnPlateau: halves LR when val_accuracy stalls (same as CIFAR-10's improved model).
- EarlyStopping: stops training when val_accuracy stops improving, restores best weights.
- ModelCheckpoint: saves the best model during training, not just the final one.
- Stratified train/validation split: keeps class distribution balanced.
- CONFIG dict: all hyperparameters in one place — easy to tune.

Key differences vs CIFAR-10's train_improved.py:
- dropout_rate default raised to 0.5 (more overfitting risk with 500 imgs/class)
- early_stop_patience raised to 10 (100 classes takes longer to converge)
- validation_split kept at 0.1 (same)
- epochs kept at 50 max (EarlyStopping will fire earlier)

How to run:
    python cifar100/src/train.py

What you will see:
- Per-epoch training progress
- Callbacks firing (LR reductions, early stop)
- Final test accuracy
- Saved model: cifar100/outputs/cnn_cifar100.keras
- Training curves: cifar100/outputs/training_curves.png
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Project root is the cifar10-project folder
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
cifar100_root = os.path.join(project_root, "cifar100")
sys.path.insert(0, project_root)

from cifar100.src.data_loader import load_cifar100
from cifar100.src.augmentation import create_datagen
from cifar100.src.model_cnn import build_cnn

CONFIG = {
    "epochs":              80,     # more epochs — cosine decay benefits from longer training
    "batch_size":          64,
    "learning_rate":       1e-3,
    "dropout_rate":        0.5,    # higher than CIFAR-10 (0.4) — more overfitting risk
    "validation_split":    0.1,
    "random_seed":         42,
    "early_stop_patience": 15,     # higher patience — cosine decay needs time to settle
    "label_smoothing":     0.1,    # softens hard 0/1 targets — helps with 100 classes
}


def split_train_validation(x_train, y_train, validation_split, seed):
    """
    Stratified split so each class is proportionally represented
    in both train and validation sets.
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
    val_indices   = np.array(val_indices)
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    return (
        x_train[train_indices], y_train[train_indices],
        x_train[val_indices],   y_train[val_indices],
    )


def build_callbacks(output_dir):
    """
    Build training callbacks.

    EarlyStopping   — stops training and restores best weights.
    ModelCheckpoint — saves only the best model seen during training.
    TensorBoard     — logs metrics; run: tensorboard --logdir=cifar100/logs

    Note: No ReduceLROnPlateau — cosine decay handles LR scheduling automatically.
    """
    log_dir = os.path.join(cifar100_root, "logs")
    os.makedirs(log_dir, exist_ok=True)

    checkpoint_path = os.path.join(output_dir, "cnn_cifar100.keras")

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=CONFIG["early_stop_patience"],
        restore_best_weights=True,
        verbose=1,
    )
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    )
    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
    )

    return [early_stopping, model_checkpoint, tensorboard]


def plot_training_curves(history, output_dir):
    """Plot and save accuracy and loss curves."""
    epochs = range(1, len(history.history["accuracy"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, history.history["accuracy"],     label="Train")
    axes[0].plot(epochs, history.history["val_accuracy"], label="Validation")
    axes[0].set_title("CIFAR-100 CNN — Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(epochs, history.history["loss"],     label="Train")
    axes[1].plot(epochs, history.history["val_loss"], label="Validation")
    axes[1].set_title("CIFAR-100 CNN — Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()
    save_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(save_path)
    plt.show()
    print(f"Training curves saved to: {save_path}")


def train():
    """Full training pipeline for CIFAR-100 CNN."""
    print("=" * 60)
    print("  Training CNN on CIFAR-100")
    print("=" * 60)
    print("\nHyperparameters:")
    for k, v in CONFIG.items():
        print(f"  {k:<24} {v}")
    print()

    # Data
    x_train_full, y_train_full, x_test, y_test, _ = load_cifar100()

    x_train, y_train, x_val, y_val = split_train_validation(
        x_train_full, y_train_full,
        validation_split=CONFIG["validation_split"],
        seed=CONFIG["random_seed"],
    )

    datagen = create_datagen()
    datagen.fit(x_train)

    # Model
    model = build_cnn(dropout_rate=CONFIG["dropout_rate"])

    # Cosine decay: LR starts at 1e-3 and smoothly decays to ~0 over all epochs.
    # Better than ReduceLROnPlateau for longer training — no sudden drops.
    steps_per_epoch = int(len(x_train) * (1 - CONFIG["validation_split"])) // CONFIG["batch_size"]
    total_steps     = CONFIG["epochs"] * steps_per_epoch
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=CONFIG["learning_rate"],
        decay_steps=total_steps,
        alpha=1e-6,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        # Label smoothing: instead of training towards hard 1.0 targets,
        # trains towards 0.9 — reduces overconfidence and improves generalisation.
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=CONFIG["label_smoothing"]),
        metrics=["accuracy"],
    )
    model.summary()

    # Callbacks
    output_dir = os.path.join(cifar100_root, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    callbacks = build_callbacks(output_dir)

    # Train
    print("\nStarting training...")
    history = model.fit(
        datagen.flow(
            x_train, y_train,
            batch_size=CONFIG["batch_size"],
            shuffle=True,
            seed=CONFIG["random_seed"],
        ),
        epochs=CONFIG["epochs"],
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
    print(f"\nCIFAR-100 CNN — Test Accuracy : {test_acc:.4f}")
    print(f"CIFAR-100 CNN — Test Loss     : {test_loss:.4f}")

    # Save results
    results = {**CONFIG, "test_accuracy": round(float(test_acc), 4), "test_loss": round(float(test_loss), 4)}
    results_path = os.path.join(output_dir, "cnn_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")

    plot_training_curves(history, output_dir)

    return model, history, test_acc


if __name__ == "__main__":
    train()
