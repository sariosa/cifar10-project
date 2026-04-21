"""
transfer_learning.py

Transfer learning for CIFAR-100 using EfficientNetB3 pretrained on ImageNet.

Why EfficientNetB3 instead of ResNet50:
-----------------------------------------
EfficientNet models are significantly more accurate than ResNet50 for the same
compute budget. EfficientNetB3 typically gets ~5-10% higher accuracy on
CIFAR-100 than ResNet50 because it uses compound scaling — it scales depth,
width, and resolution together rather than just stacking more layers.

Other improvements over the original ResNet50 script:
- Label smoothing (0.1): prevents the model from becoming overconfident on
  100 classes. Instead of training towards hard 0/1 targets, it trains towards
  0.001/0.9 which generalises better.
- Cosine decay LR schedule: smoother learning rate reduction than
  ReduceLROnPlateau — the LR follows a cosine curve from max to near-zero.
- Unfreezes top 50 layers for fine-tuning (vs 30 for ResNet50) — EfficientNetB3
  has more layers so we need to unfreeze more to adapt it.
- Memory safe: uses tf.data pipeline to resize images per batch (32 at a time)
  instead of pre-resizing all 60k images into RAM (~27GB crash).

How to run:
    python cifar100/src/transfer_learning.py

What you will see:
- Phase 1: head training (15 epochs, base frozen)
- Phase 2: fine-tuning (15 epochs, top 50 layers unfrozen)
- Final test accuracy
- Saved model: cifar100/outputs/efficientnetb3_cifar100.keras
- Training curves: cifar100/outputs/transfer_learning_curves.png
"""

import os
import sys
import json
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras import layers, models

project_root  = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
cifar100_root = os.path.join(project_root, "cifar100")
sys.path.insert(0, project_root)

from cifar100.src.data_loader import load_cifar100

TARGET_SIZE = (224, 224)
BATCH_SIZE  = 32


def make_dataset(x, y, training=False):
    """
    Build a tf.data pipeline that resizes images to 224x224 on the fly.

    EfficientNetB3 expects pixel values in [0, 255]. Our data is normalised
    to [0, 1], so we rescale back inside the pipeline.

    Args:
        x:        Image array (N, 32, 32, 3) float32 in [0, 1]
        y:        One-hot label array (N, 100)
        training: If True, shuffle the dataset each epoch.

    Returns:
        tf.data.Dataset yielding (224x224 image batch in [0,255], label batch)
    """
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if training:
        ds = ds.shuffle(buffer_size=10000, seed=42)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.map(
        lambda imgs, labels: (
            tf.image.resize(imgs * 255.0, TARGET_SIZE),  # rescale to [0,255] for EfficientNet
            labels,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def build_transfer_model(num_epochs_phase1):
    """
    Build the EfficientNetB3 transfer learning model.

    Architecture:
    - EfficientNetB3 base pretrained on ImageNet (frozen initially)
    - GlobalAveragePooling2D
    - BatchNormalization
    - Dense(512, relu)
    - Dropout(0.5)
    - Dense(100, softmax)

    Args:
        num_epochs_phase1: Used to build the cosine decay schedule for phase 1.

    Returns:
        model:      Compiled Keras model.
        base_model: EfficientNetB3 base (used later for unfreezing).
    """
    base_model = EfficientNetB3(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3),
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(100, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)

    steps_per_epoch = 45000 // BATCH_SIZE
    total_steps     = num_epochs_phase1 * steps_per_epoch

    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-3,
        decay_steps=total_steps,
        alpha=1e-6,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        # Label smoothing: softens hard targets — helps generalise with 100 classes
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )

    return model, base_model


def build_callbacks(output_dir, phase):
    """EarlyStopping + ModelCheckpoint for a training phase."""
    checkpoint_path = os.path.join(output_dir, f"efficientnetb3_cifar100_{phase}.keras")

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=6,
        restore_best_weights=True,
        verbose=1,
    )
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    )
    return [early_stopping, model_checkpoint]


def plot_training_curves(history_head, history_finetune, output_dir):
    """Plot combined training curves across both phases."""
    train_acc = history_head.history["accuracy"]     + history_finetune.history["accuracy"]
    val_acc   = history_head.history["val_accuracy"] + history_finetune.history["val_accuracy"]
    epochs      = range(1, len(train_acc) + 1)
    phase_split = len(history_head.history["accuracy"])

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_acc, label="Train Accuracy")
    plt.plot(epochs, val_acc,   label="Val Accuracy")
    plt.axvline(x=phase_split, color="gray", linestyle="--", label="Fine-tuning starts")
    plt.title("CIFAR-100 EfficientNetB3: Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(output_dir, "transfer_learning_curves.png")
    plt.savefig(save_path)
    plt.show()
    print(f"Training curves saved to: {save_path}")


def train():
    """Full transfer learning pipeline for CIFAR-100."""
    output_dir = os.path.join(cifar100_root, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    EPOCHS_PHASE1 = 15
    EPOCHS_PHASE2 = 15

    # Load raw 32x32 data — stays small in RAM
    x_train_full, y_train_full, x_test, y_test, _ = load_cifar100()

    # 10% validation split
    val_split = int(len(x_train_full) * 0.1)
    x_val,   y_val   = x_train_full[:val_split],  y_train_full[:val_split]
    x_train, y_train = x_train_full[val_split:],  y_train_full[val_split:]

    # tf.data pipelines — resize happens per batch, not all at once
    train_ds = make_dataset(x_train, y_train, training=True)
    val_ds   = make_dataset(x_val,   y_val,   training=False)
    test_ds  = make_dataset(x_test,  y_test,  training=False)

    # Build model
    model, base_model = build_transfer_model(EPOCHS_PHASE1)
    model.summary()

    # Phase 1: train head only (base frozen)
    print("\nPhase 1: Training custom head (EfficientNetB3 base frozen)...")
    history_head = model.fit(
        train_ds,
        epochs=EPOCHS_PHASE1,
        validation_data=val_ds,
        callbacks=build_callbacks(output_dir, "head"),
        verbose=1,
    )

    # Phase 2: fine-tune top 50 layers
    print("\nPhase 2: Fine-tuning top 50 layers of EfficientNetB3...")
    base_model.trainable = True
    for layer in base_model.layers[:-50]:
        layer.trainable = False

    steps_per_epoch = len(x_train) // BATCH_SIZE
    total_steps     = EPOCHS_PHASE2 * steps_per_epoch

    # Lower LR with cosine decay for fine-tuning
    lr_finetune = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-5,
        decay_steps=total_steps,
        alpha=1e-7,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_finetune),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )

    history_finetune = model.fit(
        train_ds,
        epochs=EPOCHS_PHASE2,
        validation_data=val_ds,
        callbacks=build_callbacks(output_dir, "finetune"),
        verbose=1,
    )

    # Final evaluation
    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    print(f"\nEfficientNetB3 — Test Accuracy: {test_acc:.4f}  ({test_acc*100:.1f}%)")
    print(f"EfficientNetB3 — Test Loss:     {test_loss:.4f}")

    # Save final model and results
    save_path = os.path.join(output_dir, "efficientnetb3_cifar100.keras")
    model.save(save_path)
    print(f"Model saved to: {save_path}")

    results = {
        "model":         "EfficientNetB3",
        "test_accuracy": round(float(test_acc), 4),
        "test_loss":     round(float(test_loss), 4),
        "epochs_phase1": EPOCHS_PHASE1,
        "epochs_phase2": EPOCHS_PHASE2,
        "label_smoothing": 0.1,
    }
    with open(os.path.join(output_dir, "transfer_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    plot_training_curves(history_head, history_finetune, output_dir)

    return model, test_acc


if __name__ == "__main__":
    train()
