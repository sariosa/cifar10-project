"""
transfer_learning.py

What is Transfer Learning?
--------------------------
Transfer learning means taking a model that was already trained on a large
dataset (ImageNet — 1.2 million images, 1000 classes) and reusing it for
a different task (CIFAR-10 — 60,000 images, 10 classes).

Why does it perform better than training from scratch?
------------------------------------------------------
- The pretrained model has already learned powerful low-level features
  (edges, textures, shapes) from millions of images.
- Instead of learning these from scratch, we just adapt the final layers
  to our specific 10 classes.
- This means we need less data and less training time to get high accuracy.
- It is like hiring someone who already knows how to cook and just teaching
  them a new recipe, instead of teaching someone to cook from zero.

What this file does:
--------------------
1. Loads and resizes CIFAR-10 images from 32x32 to 224x224 (required by ResNet50).
2. Loads ResNet50 pretrained on ImageNet, without the top classification layer.
3. Freezes the base model so pretrained weights are not changed during training.
4. Adds a custom classification head on top for CIFAR-10's 10 classes.
5. Trains only the new head first, then fine-tunes the whole model.
6. Evaluates and compares accuracy vs the baseline CNN.
7. Visualises and saves the model architecture.

How to run:
    python src/transfer_learning.py

What you will see:
- Training progress for each epoch
- Final test accuracy printed in the terminal
- Accuracy comparison vs baseline CNN
- Saved model at: outputs/resnet50_cifar10.keras
- Architecture diagram at: outputs/resnet50_architecture.png
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
from tensorflow.keras.utils import plot_model

# Set the project root so imports from the src folder work correctly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.data_loader import load_cifar10


# ── Step 1: Resize CIFAR-10 images ──────────────────────────────────────────

def resize_images(x, target_size=(224, 224)):
    """
    Resize images from 32x32 to 224x224 using tf.image.resize.

    ResNet50 was designed for 224x224 images, so we must upsample CIFAR-10.

    Args:
        x: Array of images, shape (N, 32, 32, 3)
        target_size: Tuple (height, width) to resize to

    Returns:
        Resized image array, shape (N, 224, 224, 3)
    """
    print(f"Resizing {len(x)} images to {target_size}...")
    x_resized = tf.image.resize(x, target_size).numpy()
    print("Resizing done.")
    return x_resized


# ── Step 2: Build the Transfer Learning model ────────────────────────────────

def build_transfer_model():
    """
    Build a transfer learning model using ResNet50 as the base.

    Architecture:
    - ResNet50 base (pretrained on ImageNet, frozen)
    - GlobalAveragePooling2D to reduce spatial dimensions
    - Dense(256, relu) — custom head
    - Dropout(0.5) — regularisation to reduce overfitting
    - Dense(10, softmax) — output layer for 10 CIFAR-10 classes

    The base is frozen first so we only train the new head.
    After initial training, we unfreeze the top layers for fine-tuning.

    Returns:
        model: Compiled Keras model
        base_model: The ResNet50 base (used later for unfreezing)
    """
    # Load ResNet50 without the top classification layer
    # include_top=False means we remove ImageNet's 1000-class output
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # Freeze all base model layers — do not change pretrained weights yet
    base_model.trainable = False

    # Build the full model
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model, base_model


# ── Step 3: Fine-tune ────────────────────────────────────────────────────────

def fine_tune(model, base_model, x_train, y_train, x_test, y_test):
    """
    Unfreeze the top layers of ResNet50 and fine-tune the whole model
    with a lower learning rate.

    Fine-tuning lets the pretrained layers slightly adjust to CIFAR-10,
    which squeezes out extra accuracy after the head is already trained.

    Args:
        model: The full model (head already trained)
        base_model: The ResNet50 base layer
        x_train, y_train: Training data
        x_test, y_test: Test data

    Returns:
        history_finetune: Training history from fine-tuning
    """
    print("\nFine-tuning: unfreezing top layers of ResNet50...")

    # Unfreeze the entire base model
    base_model.trainable = True

    # Recompile with a much lower learning rate to avoid destroying pretrained weights
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    history_finetune = model.fit(
        x_train,
        y_train,
        epochs=5,
        batch_size=32,
        validation_data=(x_test, y_test),
        verbose=1
    )

    return history_finetune


# ── Step 4: Train, evaluate, compare ────────────────────────────────────────

def train_transfer_model(x_train, y_train, x_test, y_test):
    """
    Full training pipeline:
    1. Build the transfer learning model
    2. Train the custom head (base frozen) for 10 epochs
    3. Fine-tune the top layers for 5 more epochs
    4. Evaluate on test set

    Args:
        x_train, y_train: Resized training data
        x_test, y_test: Resized test data

    Returns:
        model: Trained model
        history_head: History from head training
        history_finetune: History from fine-tuning
        test_acc: Final test accuracy
    """
    model, base_model = build_transfer_model()

    print("\nPhase 1: Training custom head (base frozen)...")
    history_head = model.fit(
        x_train,
        y_train,
        epochs=10,
        batch_size=32,
        validation_data=(x_test, y_test),
        verbose=1
    )

    # Fine-tune the model
    history_finetune = fine_tune(model, base_model, x_train, y_train, x_test, y_test)

    # Final evaluation
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
    print(f"\nTransfer Learning Test Accuracy: {test_acc:.4f}")
    print(f"Transfer Learning Test Loss:     {test_loss:.4f}")

    return model, history_head, history_finetune, test_acc


# ── Step 5: Visualise architecture ──────────────────────────────────────────

def visualise_architecture(model, output_dir):
    """
    Print the model summary and save an architecture diagram.

    Args:
        model: Trained Keras model
        output_dir: Folder to save the diagram
    """
    print("\nModel Summary:")
    model.summary()

    diagram_path = os.path.join(output_dir, "resnet50_architecture.png")
    try:
        plot_model(
            model,
            to_file=diagram_path,
            show_shapes=True,
            show_layer_names=True
        )
        print(f"Architecture diagram saved to: {diagram_path}")
    except Exception:
        print("Could not save architecture diagram (pydot/graphviz may not be installed).")


# ── Step 6: Compare accuracy ─────────────────────────────────────────────────

def compare_accuracy(acc_baseline, acc_transfer):
    """
    Print a clear accuracy comparison between baseline CNN and transfer learning model.

    Args:
        acc_baseline: Test accuracy of the baseline CNN (float)
        acc_transfer: Test accuracy of the transfer learning model (float)
    """
    print("\n--- Accuracy Comparison ---")
    print(f"Baseline CNN         : {acc_baseline:.4f}")
    print(f"Transfer Learning    : {acc_transfer:.4f}")
    diff = acc_transfer - acc_baseline
    if diff > 0:
        print(f"Transfer learning is better by {diff:.4f} ({diff*100:.2f}%)")
    else:
        print(f"Baseline is better by {abs(diff):.4f} ({abs(diff)*100:.2f}%)")


# ── Step 7: Plot training curves ─────────────────────────────────────────────

def plot_training_curves(history_head, history_finetune, output_dir):
    """
    Plot validation accuracy across both training phases (head + fine-tune).

    Args:
        history_head: History from Phase 1 (head training)
        history_finetune: History from Phase 2 (fine-tuning)
        output_dir: Folder to save the plot
    """
    # Combine both phases
    val_acc = history_head.history["val_accuracy"] + history_finetune.history["val_accuracy"]
    train_acc = history_head.history["accuracy"] + history_finetune.history["accuracy"]
    epochs = range(1, len(val_acc) + 1)
    phase_split = len(history_head.history["val_accuracy"])

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_acc, label="Train Accuracy")
    plt.plot(epochs, val_acc, label="Val Accuracy")
    plt.axvline(x=phase_split, color="gray", linestyle="--", label="Fine-tuning starts")
    plt.title("Transfer Learning: Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "transfer_learning_curves.png")
    plt.savefig(plot_path)
    plt.show()
    print(f"Training curves saved to: {plot_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    """
    Main function:
    - loads and resizes CIFAR-10 data
    - builds and trains the ResNet50 transfer learning model
    - evaluates and compares accuracy vs baseline CNN
    - visualises architecture and plots training curves
    """
    output_dir = os.path.join(project_root, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Load CIFAR-10
    x_train, y_train, x_test, y_test, class_names = load_cifar10()

    # Resize to 224x224 for ResNet50
    x_train_resized = resize_images(x_train)
    x_test_resized = resize_images(x_test)

    # Train transfer learning model
    model, history_head, history_finetune, acc_transfer = train_transfer_model(
        x_train_resized, y_train, x_test_resized, y_test
    )

    # Save the trained model
    save_path = os.path.join(output_dir, "resnet50_cifar10.keras")
    model.save(save_path)
    print(f"\nModel saved to: {save_path}")

    # Visualise architecture
    visualise_architecture(model, output_dir)

    # Plot training curves
    plot_training_curves(history_head, history_finetune, output_dir)

    # Compare vs baseline CNN
    # Baseline accuracy from train_baseline.py run — update this value after running baseline
    acc_baseline = 0.70
    compare_accuracy(acc_baseline, acc_transfer)


# Run the script only when executed directly
if __name__ == "__main__":
    main()
