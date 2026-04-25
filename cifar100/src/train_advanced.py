"""
train_advanced.py

Advanced CIFAR-100 training pipeline combining:
  - EfficientNetV2S pretrained on ImageNet (stronger backbone than EfficientNetB3)
  - CutMix augmentation applied per-batch during training
  - Cosine decay learning rate schedule in both phases
  - Label smoothing (0.1) throughout
  - EarlyStopping + ModelCheckpoint callbacks

Why this combination?
----------------------
The three individual improvements compound well:

  1. EfficientNetV2S vs EfficientNetB3:
     ~1–3% accuracy gain from a better backbone with similar compute.

  2. CutMix on top of standard augmentation:
     Forces the model to rely on distributed features rather than
     single discriminative patches. Typically +2–4% on CIFAR-100.

  3. Cosine decay + label smoothing (already in transfer_learning.py):
     Kept unchanged — they're already well-tuned.

Expected result: ~80–85% test accuracy, up from ~77% with EfficientNetB3
(depending on hardware and number of training epochs).

Training phases:
  Phase 1 — head training (base frozen, 15 epochs)
    CutMix is applied; base BatchNorm runs in inference mode.
  Phase 2 — fine-tuning (top 50 layers unfrozen, 15 epochs)
    CutMix continues; lower learning rate prevents catastrophic forgetting.

How to run:
    python cifar100/src/train_advanced.py

What you will see:
  - Per-epoch training progress (loss + accuracy)
  - Callbacks firing (early stop, checkpoint)
  - Comparison table vs baseline EfficientNetB3 result
  - Saved model:      cifar100/outputs/efficientnetv2s_advanced.keras
  - Training curves:  cifar100/outputs/advanced_training_curves.png
  - Results JSON:     cifar100/outputs/advanced_results.json
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

project_root  = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
cifar100_root = os.path.join(project_root, "cifar100")
sys.path.insert(0, project_root)

from cifar100.src.data_loader import load_cifar100
from cifar100.src.augmentation import create_datagen
from cifar100.src.augmentation_advanced import augment_batch
from cifar100.src.model_efficientnetv2 import build_model, unfreeze_top_layers


# ── Hyperparameters ───────────────────────────────────────────────────────────

CONFIG = {
    "backbone":            "EfficientNetV2S",
    "input_size":          224,          # pixels; required by EfficientNetV2S
    "batch_size":          32,           # keep small — 224x224 images use more memory
    "epochs_phase1":       15,           # head training (base frozen)
    "epochs_phase2":       15,           # fine-tuning (top 50 layers unfrozen)
    "initial_lr_phase1":   1e-3,
    "finetune_lr_phase2":  1e-5,
    "dropout_rate":        0.5,
    "label_smoothing":     0.1,
    "validation_split":    0.1,
    "random_seed":         42,
    "unfreeze_top_layers": 50,
    "cutmix_alpha":        0.4,          # Beta distribution parameter for CutMix
    "early_stop_patience": 6,
}


# ── Data helpers ──────────────────────────────────────────────────────────────

def split_train_validation(x_train, y_train, validation_split, seed):
    """
    Stratified split so each of the 100 classes is proportionally
    represented in both train and validation subsets.

    Mirrors the implementation in cifar100/src/train.py so results
    are comparable.
    """
    rng = np.random.default_rng(seed)
    class_ids = np.argmax(y_train, axis=1)

    train_indices = []
    val_indices   = []

    for class_id in range(y_train.shape[1]):
        indices = np.where(class_ids == class_id)[0]
        rng.shuffle(indices)
        n_val = int(len(indices) * validation_split)
        val_indices.extend(indices[:n_val])
        train_indices.extend(indices[n_val:])

    train_indices = np.array(train_indices)
    val_indices   = np.array(val_indices)
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    return (
        x_train[train_indices], y_train[train_indices],
        x_train[val_indices],   y_train[val_indices],
    )


def make_dataset(x, y, training=False):
    """
    Build a tf.data pipeline that resizes images to CONFIG["input_size"]
    on the fly. EfficientNetV2S expects pixel values in [0, 255].

    Mirrors the make_dataset() function in transfer_learning.py.
    CutMix is NOT applied here — it is applied in the custom training
    loop generator below, before the batch is passed to model.fit().
    """
    target = (CONFIG["input_size"], CONFIG["input_size"])
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if training:
        ds = ds.shuffle(buffer_size=10_000, seed=CONFIG["random_seed"])
    ds = ds.batch(CONFIG["batch_size"])
    ds = ds.map(
        lambda imgs, labels: (
            tf.image.resize(imgs * 255.0, target),
            labels,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def cutmix_generator(x, y, datagen, batch_size, alpha, seed):
    """
    Generator that:
      1. Applies standard ImageDataGenerator augmentation (flip, zoom, etc.)
      2. Then applies CutMix to each batch.

    This wraps around the existing create_datagen() pipeline so the
    two augmentation layers stack correctly.

    Args:
        x:          Training images, float32 [0, 1], shape (N, 32, 32, 3).
        y:          One-hot labels, shape (N, 100).
        datagen:    Fitted ImageDataGenerator from create_datagen().
        batch_size: Batch size.
        alpha:      CutMix Beta distribution parameter.
        seed:       Random seed for reproducibility.

    Yields:
        (x_aug_resized, y_aug) where x_aug_resized is a tf.Tensor
        shape (batch_size, 224, 224, 3) in [0, 255].
    """
    target_size = CONFIG["input_size"]
    flow = datagen.flow(x, y, batch_size=batch_size, shuffle=True, seed=seed)

    while True:
        x_batch, y_batch = next(flow)
        x_aug, y_aug = augment_batch(x_batch, y_batch, alpha=alpha, mode="cutmix")

        # Resize to 224x224 and scale to [0, 255] for EfficientNetV2S
        x_resized = tf.image.resize(x_aug * 255.0, (target_size, target_size))
        yield x_resized, y_aug


# ── Callbacks ─────────────────────────────────────────────────────────────────

def build_callbacks(output_dir, phase_name):
    """
    EarlyStopping + ModelCheckpoint for a named training phase.

    Mirrors the build_callbacks() pattern from transfer_learning.py.
    """
    checkpoint_path = os.path.join(
        output_dir, f"efficientnetv2s_advanced_{phase_name}.keras"
    )

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
    return [early_stopping, model_checkpoint]


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_training_curves(history_phase1, history_phase2, output_dir):
    """
    Plot combined training curves for both phases.

    A vertical dashed line marks where Phase 2 (fine-tuning) begins.
    Mirrors the plot_training_curves() function in transfer_learning.py.
    """
    train_acc = (
        history_phase1.history["accuracy"]
        + history_phase2.history["accuracy"]
    )
    val_acc = (
        history_phase1.history["val_accuracy"]
        + history_phase2.history["val_accuracy"]
    )
    epochs      = range(1, len(train_acc) + 1)
    phase_split = len(history_phase1.history["accuracy"])

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_acc, label="Train Accuracy")
    plt.plot(epochs, val_acc,   label="Val Accuracy")
    plt.axvline(
        x=phase_split,
        color="gray",
        linestyle="--",
        label="Fine-tuning starts",
    )
    plt.title("CIFAR-100 EfficientNetV2S + CutMix: Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(output_dir, "advanced_training_curves.png")
    plt.savefig(save_path)
    plt.show()
    print(f"Training curves saved to: {save_path}")


# ── Comparison ────────────────────────────────────────────────────────────────

def print_comparison(acc_advanced):
    """
    Print an accuracy comparison table against the EfficientNetB3 baseline
    stored in cifar100/outputs/transfer_results.json.

    Falls back to a hardcoded reference (0.7746) if the file is not found —
    this is the accuracy reported in the README for the EfficientNetB3 run.
    """
    baseline_path = os.path.join(cifar100_root, "outputs", "transfer_results.json")
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            acc_baseline = json.load(f).get("test_accuracy", 0.7746)
    else:
        acc_baseline = 0.7746

    diff = acc_advanced - acc_baseline
    direction = "improvement" if diff >= 0 else "regression"

    print("\n" + "=" * 55)
    print("  ACCURACY COMPARISON")
    print("=" * 55)
    print(f"  EfficientNetB3 (baseline) : {acc_baseline:.4f}  ({acc_baseline*100:.1f}%)")
    print(f"  EfficientNetV2S + CutMix  : {acc_advanced:.4f}  ({acc_advanced*100:.1f}%)")
    print(f"  Difference                : {diff:+.4f}  ({diff*100:+.1f}%)  ← {direction}")
    print("=" * 55)


# ── Main training pipeline ────────────────────────────────────────────────────

def train():
    """
    Full two-phase training pipeline:

    Phase 1 — Head training
      - Base (EfficientNetV2S) is frozen.
      - Only the custom Dense head is trained.
      - CutMix + standard augmentation applied per batch.
      - 15 epochs max with EarlyStopping.

    Phase 2 — Fine-tuning
      - Top 50 layers of EfficientNetV2S are unfrozen.
      - Very low learning rate (1e-5) with cosine decay.
      - CutMix continues.
      - 15 epochs max with EarlyStopping.
    """
    print("=" * 60)
    print("  Advanced CIFAR-100 Training: EfficientNetV2S + CutMix")
    print("=" * 60)
    print("\nHyperparameters:")
    for k, v in CONFIG.items():
        print(f"  {k:<26} {v}")
    print()

    output_dir = os.path.join(cifar100_root, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────────────────
    x_train_full, y_train_full, x_test, y_test, _ = load_cifar100()

    x_train, y_train, x_val, y_val = split_train_validation(
        x_train_full,
        y_train_full,
        validation_split=CONFIG["validation_split"],
        seed=CONFIG["random_seed"],
    )

    print(f"Training samples:   {len(x_train)}")
    print(f"Validation samples: {len(x_val)}")
    print(f"Test samples:       {len(x_test)}")

    # Standard augmentation (flip, zoom, shift, shear) from existing pipeline
    datagen = create_datagen()
    datagen.fit(x_train)

    # Validation and test sets use the resize-only tf.data pipeline
    # (no augmentation on validation/test — standard practice)
    val_ds  = make_dataset(x_val,  y_val,  training=False)
    test_ds = make_dataset(x_test, y_test, training=False)

    steps_per_epoch = len(x_train) // CONFIG["batch_size"]

    # ── Model ─────────────────────────────────────────────────────────────────
    model, base_model = build_model(
        num_epochs_phase1=CONFIG["epochs_phase1"],
        steps_per_epoch=steps_per_epoch,
        dropout_rate=CONFIG["dropout_rate"],
        initial_lr=CONFIG["initial_lr_phase1"],
        label_smoothing=CONFIG["label_smoothing"],
    )
    model.summary()

    # ── Phase 1: head training ────────────────────────────────────────────────
    print("\nPhase 1: Training custom head (EfficientNetV2S base frozen)...")

    # Use the CutMix generator for training; validation uses the plain tf.data pipeline
    train_gen = cutmix_generator(
        x_train, y_train,
        datagen=datagen,
        batch_size=CONFIG["batch_size"],
        alpha=CONFIG["cutmix_alpha"],
        seed=CONFIG["random_seed"],
    )

    history_phase1 = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=CONFIG["epochs_phase1"],
        validation_data=val_ds,
        callbacks=build_callbacks(output_dir, "head"),
        verbose=1,
    )

    # ── Phase 2: fine-tuning ──────────────────────────────────────────────────
    print(f"\nPhase 2: Fine-tuning top {CONFIG['unfreeze_top_layers']} layers of EfficientNetV2S...")

    unfreeze_top_layers(
        model,
        base_model,
        n_layers=CONFIG["unfreeze_top_layers"],
        num_epochs_phase2=CONFIG["epochs_phase2"],
        steps_per_epoch=steps_per_epoch,
        finetune_lr=CONFIG["finetune_lr_phase2"],
        label_smoothing=CONFIG["label_smoothing"],
    )

    # Re-create the generator (the flow object is stateful, so start fresh)
    train_gen = cutmix_generator(
        x_train, y_train,
        datagen=datagen,
        batch_size=CONFIG["batch_size"],
        alpha=CONFIG["cutmix_alpha"],
        seed=CONFIG["random_seed"] + 1,
    )

    history_phase2 = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=CONFIG["epochs_phase2"],
        validation_data=val_ds,
        callbacks=build_callbacks(output_dir, "finetune"),
        verbose=1,
    )

    # ── Final evaluation ──────────────────────────────────────────────────────
    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    print(f"\nEfficientNetV2S + CutMix — Test Accuracy: {test_acc:.4f}  ({test_acc*100:.1f}%)")
    print(f"EfficientNetV2S + CutMix — Test Loss:     {test_loss:.4f}")

    # ── Save artefacts ────────────────────────────────────────────────────────
    save_path = os.path.join(output_dir, "efficientnetv2s_advanced.keras")
    model.save(save_path)
    print(f"\nModel saved to: {save_path}")

    results = {
        **CONFIG,
        "test_accuracy": round(float(test_acc), 4),
        "test_loss":     round(float(test_loss), 4),
    }
    results_path = os.path.join(output_dir, "advanced_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")

    plot_training_curves(history_phase1, history_phase2, output_dir)
    print_comparison(test_acc)

    return model, test_acc


# Run the script only when executed directly
if __name__ == "__main__":
    train()