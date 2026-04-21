"""
predict.py

Run inference on a single image using a trained CIFAR-100 model.

How to run:
    python cifar100/src/predict.py --image path/to/image.jpg
    python cifar100/src/predict.py --image path/to/image.jpg --model cifar100/outputs/resnet50_cifar100.keras

What you will see:
- Top-5 predicted classes with confidence scores
- A bar chart saved to cifar100/outputs/prediction.png
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

project_root  = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
cifar100_root = os.path.join(project_root, "cifar100")
sys.path.insert(0, project_root)

from cifar100.src.data_loader import CIFAR100_CLASSES


def load_image(image_path, target_size):
    """
    Load and preprocess an image for inference.

    Args:
        image_path:  Path to the image file.
        target_size: (height, width) expected by the model.

    Returns:
        Preprocessed image array, shape (1, H, W, 3).
    """
    img = tf.keras.utils.load_img(image_path, target_size=target_size)
    img_array = tf.keras.utils.img_to_array(img).astype("float32") / 255.0
    return np.expand_dims(img_array, axis=0)


def predict(model_path, image_path, top_k=5):
    """
    Load a model and predict the class of an image.

    Args:
        model_path: Path to a saved .keras model.
        image_path: Path to the input image.
        top_k:      Number of top predictions to display.
    """
    # Determine input size from model
    model = tf.keras.models.load_model(model_path)
    input_shape = model.input_shape[1:3]  # (H, W)
    print(f"Loaded model: {model_path}")
    print(f"Input size:   {input_shape}")

    # Load and preprocess image
    img = load_image(image_path, target_size=input_shape)

    # Predict
    predictions = model.predict(img, verbose=0)[0]

    # Top-k results
    top_indices = np.argsort(predictions)[::-1][:top_k]
    top_classes = [CIFAR100_CLASSES[i] for i in top_indices]
    top_scores  = [predictions[i] for i in top_indices]

    print(f"\nTop-{top_k} predictions for: {os.path.basename(image_path)}")
    print("-" * 40)
    for cls, score in zip(top_classes, top_scores):
        print(f"  {cls:<20} {score * 100:.2f}%")

    # Bar chart
    plt.figure(figsize=(8, 4))
    bars = plt.barh(top_classes[::-1], [s * 100 for s in top_scores[::-1]], color="steelblue")
    plt.xlabel("Confidence (%)")
    plt.title(f"CIFAR-100 Prediction: {os.path.basename(image_path)}")
    plt.xlim(0, 100)

    for bar, score in zip(bars, top_scores[::-1]):
        plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                 f"{score * 100:.1f}%", va="center")

    plt.tight_layout()
    output_dir = os.path.join(cifar100_root, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "prediction.png")
    plt.savefig(save_path)
    plt.show()
    print(f"\nPrediction chart saved to: {save_path}")


def main():
    default_model = os.path.join(cifar100_root, "outputs", "cnn_cifar100.keras")

    parser = argparse.ArgumentParser(description="CIFAR-100 inference")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--model", default=default_model, help="Path to .keras model file")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top predictions")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Model not found: {args.model}")
        print("Train the model first: python cifar100/src/train.py")
        sys.exit(1)

    if not os.path.exists(args.image):
        print(f"Image not found: {args.image}")
        sys.exit(1)

    predict(args.model, args.image, top_k=args.top_k)


if __name__ == "__main__":
    main()
