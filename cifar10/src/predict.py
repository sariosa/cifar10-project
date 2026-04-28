"""
predict.py

What this file does:
--------------------
Loads a saved CIFAR-10 model and predicts the class of a given image.
Works with any model saved in the outputs/ folder.

Why this file exists:
---------------------
After training, you need a way to actually use the model on new images.
This file shows the full pipeline end to end — from loading an image
to getting a predicted class label with a confidence score.

How to run:
    python src/predict.py --model outputs/CNN.keras --image path/to/image.png

    or use the default test image from CIFAR-10:
    python src/predict.py --model outputs/CNN.keras

What you will see:
- The predicted class label
- The confidence score (how sure the model is)
- A plot of the image with the prediction as the title
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.data_loader import load_cifar10

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def load_and_preprocess_image(image_path):
    """
    Load an external image file and preprocess it for prediction.

    Resizes to 32x32 and normalises pixel values to [0, 1].

    Args:
        image_path: Path to the image file

    Returns:
        img_array: Preprocessed image array of shape (1, 32, 32, 3)
        img_display: Original image for display
    """
    img = tf.keras.utils.load_img(image_path, target_size=(32, 32))
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    img_display = img_array.copy()
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, img_display


def predict(model, img_array, class_names=CIFAR10_CLASSES):
    """
    Run prediction on a preprocessed image.

    Args:
        model: Loaded Keras model
        img_array: Preprocessed image array of shape (1, 32, 32, 3)
        class_names: List of class label strings

    Returns:
        predicted_class: Predicted class name
        confidence: Confidence score (0 to 1)
    """
    predictions = model.predict(img_array, verbose=0)
    predicted_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_index]
    predicted_class = class_names[predicted_index]
    return predicted_class, confidence


def show_prediction(img_display, predicted_class, confidence):
    """
    Display the image alongside a confidence bar chart for all 10 classes.

    Left: the image with predicted label and confidence as title.
    Right: horizontal bar chart showing the model's confidence for every class.

    Args:
        img_display: Image array to display
        predicted_class: Predicted class name
        confidence: Confidence score of the top prediction
    """
    plt.figure(figsize=(4, 4))
    plt.imshow(img_display)
    plt.title(f"Predicted: {predicted_class} ({confidence * 100:.1f}%)", fontsize=13)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def show_confidence_chart(model, img_array, class_names=CIFAR10_CLASSES):
    """
    Show a horizontal bar chart of the model's confidence scores for all 10 classes.

    This makes it easy to see not just what the model predicted, but how
    confident it was across every class — useful for spotting close calls.

    Args:
        model: Loaded Keras model
        img_array: Preprocessed image array of shape (1, 32, 32, 3)
        class_names: List of CIFAR-10 class names
    """
    predictions = model.predict(img_array, verbose=0)[0]
    predicted_index = np.argmax(predictions)

    colors = ["#4CAF50" if i == predicted_index else "#90CAF9" for i in range(len(class_names))]

    plt.figure(figsize=(7, 4))
    bars = plt.barh(class_names, predictions * 100, color=colors)
    plt.xlabel("Confidence (%)")
    plt.title("Prediction Confidence per Class")
    plt.xlim(0, 100)

    # Add percentage labels on each bar
    for bar, score in zip(bars, predictions):
        plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                 f"{score * 100:.1f}%", va="center", fontsize=8)

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Predict CIFAR-10 class for an image.")
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.join(project_root, "outputs", "CNN.keras"),
        help="Path to the saved Keras model"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to an image file. If not provided, uses a sample from CIFAR-10 test set."
    )
    args = parser.parse_args()

    # Load the model
    if not os.path.exists(args.model):
        print(f"Model not found at: {args.model}")
        print("Please train the model first by running: python src/train_baseline.py")
        return

    print(f"Loading model from: {args.model}")
    model = load_model(args.model)

    # Load image
    if args.image:
        # Use provided image file
        img_array, img_display = load_and_preprocess_image(args.image)
    else:
        # Use a random sample from CIFAR-10 test set
        print("No image provided — using a random sample from CIFAR-10 test set.")
        _, _, x_test, y_test, class_names = load_cifar10()
        idx = np.random.randint(0, len(x_test))
        img_array = np.expand_dims(x_test[idx], axis=0)
        img_display = x_test[idx]
        actual_class = class_names[np.argmax(y_test[idx])]
        print(f"Actual class: {actual_class}")

    # Predict
    predicted_class, confidence = predict(model, img_array)
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence:      {confidence * 100:.1f}%")

    # Show image with prediction title
    show_prediction(img_display, predicted_class, confidence)

    # Show confidence bar chart for all classes
    show_confidence_chart(model, img_array)


if __name__ == "__main__":
    main()
