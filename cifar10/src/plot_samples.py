"""
plot_samples.py

What this file does:
--------------------
Displays a grid of sample images from the CIFAR-10 dataset with their
class labels. This is useful for understanding what the data looks like
before training any model.

Why this file exists:
---------------------
Exploring the data visually is an important first step in any ML project.
It helps confirm the data loaded correctly and gives an idea of how
challenging the classification task is (e.g. some CIFAR-10 classes
like cat vs dog or automobile vs truck look very similar at 32x32).

How to run:
    python src/plot_samples.py

What you will see:
- A 5x10 grid showing 2 sample images per class
- Each image labelled with its class name
- Plot saved to outputs/cifar10_samples.png
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.data_loader import load_cifar10


def plot_samples(x_train, y_train, class_names, samples_per_class=2):
    """
    Display a grid of sample images from each CIFAR-10 class.

    Args:
        x_train: Training images
        y_train: One-hot encoded training labels
        class_names: List of CIFAR-10 class names
        samples_per_class: Number of sample images to show per class
    """
    num_classes = len(class_names)
    fig, axes = plt.subplots(
        samples_per_class, num_classes,
        figsize=(num_classes * 1.5, samples_per_class * 1.5)
    )

    fig.suptitle("CIFAR-10 Sample Images", fontsize=14, fontweight="bold", y=1.02)

    for class_idx, class_name in enumerate(class_names):
        # Find all training images belonging to this class
        class_indices = np.where(y_train.argmax(axis=1) == class_idx)[0]

        for sample_idx in range(samples_per_class):
            ax = axes[sample_idx][class_idx]
            img = x_train[class_indices[sample_idx]]
            ax.imshow(img)
            ax.axis("off")

            # Only label the top row
            if sample_idx == 0:
                ax.set_title(class_name, fontsize=8)

    plt.tight_layout()

    output_dir = os.path.join(project_root, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "cifar10_samples.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    print(f"Sample grid saved to: {save_path}")


def main():
    x_train, y_train, _, _, class_names = load_cifar10()
    plot_samples(x_train, y_train, class_names)


if __name__ == "__main__":
    main()
