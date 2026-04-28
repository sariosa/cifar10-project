"""
augmentation_advanced.py

Advanced data augmentation for CIFAR-100: MixUp and CutMix.

Why these techniques?
----------------------
CIFAR-100 has only 500 images per class. Standard augmentation (flips,
rotations, zoom) creates visually similar variants of the same image.
MixUp and CutMix go further — they create entirely new training examples
by blending two images together, which forces the model to learn softer,
more generalised decision boundaries.

MixUp (Zhang et al., 2018):
  Creates a weighted pixel-level blend of two images and their labels.
  Example: 60% cat + 40% dog → the label becomes [0.6, 0.4] for those
  two classes. Penalises overconfidence and improves calibration.

CutMix (Yun et al., 2019):
  Cuts a rectangular patch from one image and pastes it into another.
  Labels are mixed in proportion to the cut area. Preserves local
  structure better than MixUp and is generally stronger on CIFAR-100.

Both are applied per-batch during training, not per-image at load time.
They work on top of the existing ImageDataGenerator pipeline.

How to use:
    from cifar100.src.augmentation_advanced import mixup_batch, cutmix_batch, augment_batch

    # In your training loop, after getting a batch from datagen:
    x_batch, y_batch = next(datagen_flow)
    x_aug, y_aug = augment_batch(x_batch, y_batch, alpha=0.4, mode="cutmix")
"""

import numpy as np


# ── MixUp ─────────────────────────────────────────────────────────────────────

def mixup_batch(x: np.ndarray, y: np.ndarray, alpha: float = 0.4):
    """
    Apply MixUp augmentation to a batch of images.

    For each image i in the batch, a mixing coefficient λ ~ Beta(alpha, alpha)
    is sampled and a partner image j is chosen randomly. The result is:
        x_mixed  = λ * x[i] + (1 - λ) * x[j]
        y_mixed  = λ * y[i] + (1 - λ) * y[j]

    Args:
        x:     Image batch, shape (batch_size, H, W, C), float32 in [0, 1].
        y:     One-hot label batch, shape (batch_size, num_classes).
        alpha: Beta distribution concentration parameter.
               Higher alpha → more mixing (λ closer to 0.5).
               Lower alpha → less mixing (λ closer to 0 or 1).
               Recommended: 0.2–0.4 for CIFAR-100.

    Returns:
        x_mixed: Augmented image batch, same shape as x.
        y_mixed: Soft label batch, same shape as y.
    """
    batch_size = x.shape[0]

    # Sample mixing coefficients from Beta distribution
    lambdas = np.random.beta(alpha, alpha, size=batch_size).astype(np.float32)

    # Reshape for broadcasting: (batch_size, 1, 1, 1) for images
    lam_x = lambdas.reshape(-1, 1, 1, 1)
    lam_y = lambdas.reshape(-1, 1)

    # Random permutation of the batch to select partner images
    permuted_indices = np.random.permutation(batch_size)

    x_mixed = lam_x * x + (1.0 - lam_x) * x[permuted_indices]
    y_mixed = lam_y * y + (1.0 - lam_y) * y[permuted_indices]

    return x_mixed, y_mixed


# ── CutMix ────────────────────────────────────────────────────────────────────

def _random_bounding_box(height: int, width: int, lam: float):
    """
    Compute a random bounding box for CutMix whose area fraction equals (1 - lam).

    The box dimensions are sampled so that:
        (cut_h * cut_w) / (height * width) ≈ (1 - lam)

    Args:
        height: Image height in pixels.
        width:  Image width in pixels.
        lam:    Mixing coefficient sampled from Beta(alpha, alpha).
                Determines the area of the patch.

    Returns:
        (x1, y1, x2, y2): Integer pixel coordinates of the cut rectangle.
    """
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(height * cut_ratio)
    cut_w = int(width  * cut_ratio)

    # Random centre point
    cx = np.random.randint(width)
    cy = np.random.randint(height)

    x1 = np.clip(cx - cut_w // 2, 0, width)
    y1 = np.clip(cy - cut_h // 2, 0, height)
    x2 = np.clip(cx + cut_w // 2, 0, width)
    y2 = np.clip(cy + cut_h // 2, 0, height)

    return x1, y1, x2, y2


def cutmix_batch(x: np.ndarray, y: np.ndarray, alpha: float = 0.4):
    """
    Apply CutMix augmentation to a batch of images.

    For each image i:
        1. Sample λ ~ Beta(alpha, alpha).
        2. Pick a random partner image j.
        3. Cut a rectangular patch from j and paste it into i.
        4. Adjust λ to reflect the actual area of the pasted patch.
        5. Mix labels: y_mixed = λ_adjusted * y[i] + (1 - λ_adjusted) * y[j]

    Args:
        x:     Image batch, shape (batch_size, H, W, C), float32 in [0, 1].
        y:     One-hot label batch, shape (batch_size, num_classes).
        alpha: Beta distribution parameter. Recommended: 0.4–1.0 for CIFAR-100.

    Returns:
        x_cut: Augmented image batch, same shape as x.
        y_cut: Soft label batch, same shape as y.
    """
    batch_size, height, width, channels = x.shape

    x_cut = x.copy()
    y_cut = y.copy().astype(np.float32)

    permuted_indices = np.random.permutation(batch_size)

    for i in range(batch_size):
        lam = np.random.beta(alpha, alpha)
        j   = permuted_indices[i]

        x1, y1, x2, y2 = _random_bounding_box(height, width, lam)

        # Paste the patch from partner image j into image i
        x_cut[i, y1:y2, x1:x2, :] = x[j, y1:y2, x1:x2, :]

        # Recalculate λ based on the actual patch area (may differ due to clipping)
        patch_area = (x2 - x1) * (y2 - y1)
        lam_adjusted = 1.0 - patch_area / (height * width)

        y_cut[i] = lam_adjusted * y[i] + (1.0 - lam_adjusted) * y[j]

    return x_cut, y_cut


# ── Unified interface ─────────────────────────────────────────────────────────

def augment_batch(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.4,
    mode: str = "cutmix",
    cutmix_prob: float = 0.5,
):
    """
    Apply MixUp or CutMix to a batch, with an option to randomly
    choose between the two (mode="random").

    Args:
        x:            Image batch, shape (batch_size, H, W, C), float32 in [0, 1].
        y:            One-hot label batch, shape (batch_size, num_classes).
        alpha:        Beta distribution parameter (shared by both methods).
        mode:         "cutmix"  — always apply CutMix  (recommended for CIFAR-100)
                      "mixup"   — always apply MixUp
                      "random"  — randomly choose one per batch
        cutmix_prob:  Probability of choosing CutMix when mode="random".

    Returns:
        Augmented (x, y) with the same shapes as the inputs.
    """
    if mode == "cutmix":
        return cutmix_batch(x, y, alpha=alpha)
    elif mode == "mixup":
        return mixup_batch(x, y, alpha=alpha)
    elif mode == "random":
        if np.random.random() < cutmix_prob:
            return cutmix_batch(x, y, alpha=alpha)
        else:
            return mixup_batch(x, y, alpha=alpha)
    else:
        raise ValueError(f"Unknown augmentation mode: '{mode}'. Choose 'cutmix', 'mixup', or 'random'.")