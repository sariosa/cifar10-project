import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from tensorflow.keras.applications.resnet50 import preprocess_input

# Set project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.data_loader import load_cifar10


class CompatibleBatchNormalization(tf.keras.layers.BatchNormalization):
    """
    Compatibility wrapper for older saved Keras models.

    Some models saved with older TensorFlow/Keras versions include BatchNormalization
    arguments that newer versions may not accept, such as:
    - renorm
    - renorm_clipping
    - renorm_momentum

    This class removes those unsupported arguments before loading the layer.
    """
    def __init__(self, *args, **kwargs):
        kwargs.pop("renorm", None)
        kwargs.pop("renorm_clipping", None)
        kwargs.pop("renorm_momentum", None)
        super().__init__(*args, **kwargs)


CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

AUTOTUNE = tf.data.AUTOTUNE
TRANSFER_IMAGE_SIZE = (224, 224)
TRANSFER_BATCH_SIZE = 32


def get_model_config(model_type):
    """
    Return the correct model path and settings depending on model type.

    This lets one evaluation file work for:
    - baseline CNN
    - improved CNN v3
    - transfer learning ResNet50
    """
    model_type = model_type.lower()

    if model_type == "baseline":
        return {
            "model_path": os.path.join(project_root, "outputs", "CNN.keras"),
            "display_name": "baseline",
            "needs_transfer_preprocessing": False
        }

    elif model_type == "improved":
        return {
            # This uses the patched V3 model.
            # The original V3 model was saved with older BatchNormalization config keys.
            "model_path": os.path.join(project_root, "outputs", "CNN_improved_v3_final_patched.keras"),
            "display_name": "improved_v3",
            "needs_transfer_preprocessing": False
        }

    elif model_type == "transfer":
        return {
            "model_path": os.path.join(project_root, "outputs", "resnet50_cifar10.keras"),
            "display_name": "transfer",
            "needs_transfer_preprocessing": True
        }

    else:
        raise ValueError("model_type must be one of: baseline, improved, transfer")


def load_model_with_compatibility(model_path):
    """
    Load a Keras model.

    First tries normal loading.
    If that fails because of version differences, tries compatibility mode.
    If BatchNormalization has old unsupported arguments, use a compatible
    BatchNormalization class that ignores those arguments.
    """
    try:
        model = tf.keras.models.load_model(model_path)
        return model

    except Exception as e:
        print("\nNormal model loading failed.")
        print("Trying compatibility mode...")
        print("\nOriginal loading error:")
        print(e)

        try:
            model = tf.keras.models.load_model(
                model_path,
                compile=False,
                safe_mode=False
            )
            return model

        except Exception as e:
            print("\nCompatibility mode failed.")
            print("Trying custom BatchNormalization compatibility loader...")
            print("\nSecond loading error:")
            print(e)

            model = tf.keras.models.load_model(
                model_path,
                compile=False,
                safe_mode=False,
                custom_objects={
                    "BatchNormalization": CompatibleBatchNormalization
                }
            )

            return model


def load_trained_model(model_type):
    """
    Step 1: Load the chosen model.

    Depending on the command used in the terminal, this function loads:
    - outputs/CNN.keras for baseline
    - outputs/CNN_improved_v3_final_patched.keras for improved CNN v3
    - outputs/resnet50_cifar10.keras for transfer learning
    """
    config = get_model_config(model_type)
    model_path = config["model_path"]

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at: {model_path}\n\n"
            f"If you are evaluating improved V3, first run:\n"
            f"py patch_v3_model.py\n\n"
            f"Make sure the chosen model exists in the outputs folder."
        )

    print(f"Loading {model_type} model from: {model_path}")

    model = load_model_with_compatibility(model_path)

    # Build/call the model once with a dummy input.
    # Baseline and improved expect 32x32 images.
    # Transfer learning expects 224x224 images.
    if config["needs_transfer_preprocessing"]:
        dummy_input = tf.zeros((1, 224, 224, 3), dtype=tf.float32)
    else:
        dummy_input = tf.zeros((1, 32, 32, 3), dtype=tf.float32)

    _ = model(dummy_input, training=False)

    return model, config


def preprocess_transfer_image(image, label=None):
    """
    Special preprocessing for transfer learning only.

    ResNet50 needs 224x224 images and ImageNet-style preprocessing.
    Baseline and improved CNN do NOT use this function.
    """
    image = tf.image.resize(image, TRANSFER_IMAGE_SIZE)
    image = image * 255.0
    image = preprocess_input(image)

    if label is None:
        return image
    return image, label


def create_transfer_dataset(x, y=None, batch_size=TRANSFER_BATCH_SIZE):
    """
    Create a memory-safe dataset for transfer learning evaluation.

    This avoids resizing all test images at once.
    """
    if y is None:
        ds = tf.data.Dataset.from_tensor_slices(x)
        ds = ds.map(lambda img: preprocess_transfer_image(img), num_parallel_calls=AUTOTUNE)
    else:
        ds = tf.data.Dataset.from_tensor_slices((x, y))
        ds = ds.map(preprocess_transfer_image, num_parallel_calls=AUTOTUNE)

    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


def evaluate_model(model, x_test, y_test, model_type):
    """
    Step 3: Evaluate test accuracy and test loss.
    Step 4: Predict classes.

    For baseline and improved:
    - use the original 32x32 CIFAR-10 test images.

    For transfer learning:
    - resize and preprocess the test images to 224x224 using a dataset pipeline.
    """
    if model_type == "transfer":
        test_dataset = create_transfer_dataset(x_test, y_test)

        # Step 3: Evaluate test accuracy/loss
        test_loss, test_acc = model.evaluate(test_dataset, verbose=1)
        print(f"\nTest Accuracy: {test_acc:.4f}")
        print(f"Test Loss:     {test_loss:.4f}")

        # Step 4: Predict classes
        y_prob = model.predict(test_dataset, verbose=1)

    else:
        # Step 3: Evaluate test accuracy/loss
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
        print(f"\nTest Accuracy: {test_acc:.4f}")
        print(f"Test Loss:     {test_loss:.4f}")

        # Step 4: Predict classes
        y_prob = model.predict(x_test, verbose=1)

    y_pred = np.argmax(y_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)

    return test_loss, test_acc, y_true, y_pred, y_prob


def plot_confusion_matrix(y_true, y_pred, class_names, prefix):
    """
    Step 5: Create and save the confusion matrix.

    The confusion matrix shows which classes the model predicts correctly
    and which classes it confuses with each other.
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(10, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)

    plt.title(f"Confusion Matrix - {prefix.capitalize()} - CIFAR-10")
    plt.tight_layout()

    output_dir = os.path.join(project_root, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    save_path = os.path.join(output_dir, f"{prefix}_confusion_matrix.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()

    print(f"Confusion matrix saved to: {save_path}")


def print_classification_report_file(y_true, y_pred, class_names, prefix):
    """
    Step 6: Create and save the classification report.

    The classification report includes:
    - precision
    - recall
    - F1-score
    - support

    for each CIFAR-10 class.
    """
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)

    print("\nClassification Report:\n")
    print(report)

    output_dir = os.path.join(project_root, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    save_path = os.path.join(output_dir, f"{prefix}_classification_report.txt")

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Classification report saved to: {save_path}")


def show_prediction_examples(x_test, y_true, y_pred, y_prob, class_names, prefix, correct=True, num_examples=5):
    """
    Step 7: Show and save correct predictions.
    Step 8: Show and save incorrect predictions.

    If correct=True:
    - saves examples where the model predicted correctly.

    If correct=False:
    - saves examples where the model made mistakes.
    """
    if correct:
        indices = np.where(y_true == y_pred)[0]
        title_text = "Correct Predictions"
        save_name = f"{prefix}_correct_predictions.png"
    else:
        indices = np.where(y_true != y_pred)[0]
        title_text = "Incorrect Predictions"
        save_name = f"{prefix}_incorrect_predictions.png"

    if len(indices) == 0:
        print(f"No {title_text.lower()} found.")
        return

    chosen = indices[:num_examples]

    plt.figure(figsize=(15, 3))

    for i, idx in enumerate(chosen):
        plt.subplot(1, num_examples, i + 1)
        plt.imshow(x_test[idx])

        pred_class = class_names[y_pred[idx]]
        true_class = class_names[y_true[idx]]
        confidence = y_prob[idx][y_pred[idx]] * 100

        plt.title(
            f"Pred: {pred_class}\nTrue: {true_class}\n{confidence:.1f}%",
            fontsize=10
        )
        plt.axis("off")

    plt.suptitle(f"{title_text} - {prefix.capitalize()}", fontsize=18)
    plt.tight_layout()

    output_dir = os.path.join(project_root, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    save_path = os.path.join(output_dir, save_name)
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()

    print(f"{title_text} figure saved to: {save_path}")


def find_last_conv_layer_name(model, model_type):
    """
    Helper function for Step 9: Grad-CAM.

    Grad-CAM needs the last convolutional layer.

    For baseline/improved:
    - the Conv2D layers are directly inside the model.

    For transfer learning:
    - the Conv2D layers are inside the nested ResNet50 base model.
    """
    if model_type == "transfer":
        base_model = None

        for layer in model.layers:
            if isinstance(layer, tf.keras.Model):
                base_model = layer
                break

        if base_model is None:
            raise ValueError("Could not find nested base model inside transfer model.")

        for layer in reversed(base_model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                return base_model, layer.name

        raise ValueError("No Conv2D layer found inside transfer base model.")

    else:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                return None, layer.name

        raise ValueError("No Conv2D layer found in model.")


def make_gradcam_heatmap(img_array, model, model_type, pred_index=None):
    """
    Step 9: Create Grad-CAM heatmap.

    Grad-CAM helps explain what part of the image the model focused on
    when making its prediction.
    """
    if model_type == "transfer":
        base_model, last_conv_layer_name = find_last_conv_layer_name(model, model_type)

        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[
                base_model.get_layer(last_conv_layer_name).output,
                model.output
            ]
        )

    else:
        _, last_conv_layer_name = find_last_conv_layer_name(model, model_type)

        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[
                model.get_layer(last_conv_layer_name).output,
                model.output
            ]
        )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array, training=False)

        if pred_index is None:
            pred_index = tf.argmax(predictions[0])

        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    if grads is None:
        raise ValueError("Gradients could not be computed for Grad-CAM.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)

    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy()


def prepare_single_image_for_model(img, model_type):
    """
    Helper function for Step 9: Grad-CAM.

    Baseline/improved:
    - image stays 32x32.

    Transfer learning:
    - image is resized to 224x224 and preprocessed for ResNet50.
    """
    if model_type == "transfer":
        img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
        img_tensor = tf.image.resize(img_tensor, TRANSFER_IMAGE_SIZE)
        img_tensor = img_tensor * 255.0
        img_tensor = preprocess_input(img_tensor)

        img_array = np.expand_dims(img_tensor.numpy(), axis=0).astype("float32")

    else:
        img_array = np.expand_dims(img, axis=0).astype("float32")

    return img_array


def display_gradcam(x_test, y_true, y_pred, model, class_names, model_type, prefix, num_examples=5):
    """
    Step 9: Try Grad-CAM.
    Step 10: Save Grad-CAM outputs.

    If Grad-CAM fails for one model, the script prints the error
    but continues instead of crashing completely.
    """
    try:
        chosen = np.arange(min(num_examples, len(x_test)))

        plt.figure(figsize=(15, 6))

        for i, idx in enumerate(chosen):
            img = x_test[idx]
            img_array = prepare_single_image_for_model(img, model_type)

            heatmap = make_gradcam_heatmap(
                img_array,
                model,
                model_type,
                pred_index=y_pred[idx]
            )

            plt.subplot(2, num_examples, i + 1)
            plt.imshow(img)
            plt.title(
                f"True: {class_names[y_true[idx]]}\nPred: {class_names[y_pred[idx]]}",
                fontsize=9
            )
            plt.axis("off")

            plt.subplot(2, num_examples, num_examples + i + 1)
            plt.imshow(img)
            plt.imshow(heatmap, cmap="jet", alpha=0.4)
            plt.title("Grad-CAM", fontsize=9)
            plt.axis("off")

        plt.tight_layout()

        output_dir = os.path.join(project_root, "outputs")
        os.makedirs(output_dir, exist_ok=True)

        save_path = os.path.join(output_dir, f"{prefix}_gradcam_examples.png")
        plt.savefig(save_path, bbox_inches="tight")
        plt.show()

        print(f"Grad-CAM examples saved to: {save_path}")

    except Exception as e:
        print(f"Grad-CAM could not be generated for {prefix}: {e}")


def print_confusion_analysis(y_true, y_pred, class_names, top_n=5):
    """
    Extra evaluation analysis.

    This prints the most common mistakes, for example:
    True cat -> Pred dog

    This is useful for writing the report.
    """
    cm = confusion_matrix(y_true, y_pred)
    confusion_pairs = []

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                confusion_pairs.append((cm[i, j], class_names[i], class_names[j]))

    confusion_pairs.sort(reverse=True, key=lambda x: x[0])

    print(f"\nTop {top_n} most common confusions:")

    for count, true_class, pred_class in confusion_pairs[:top_n]:
        print(f"  True {true_class:>10} -> Pred {pred_class:<10} : {count}")


def main():
    """
    Full evaluation pipeline.

    Same steps for baseline, improved v3, and transfer:

    Step 1: Load the chosen model.
    Step 2: Load CIFAR-10 test data.
    Step 3: Evaluate test accuracy/loss.
    Step 4: Predict classes.
    Step 5: Create confusion matrix.
    Step 6: Create classification report.
    Step 7: Show correct predictions.
    Step 8: Show incorrect predictions.
    Step 9: Try Grad-CAM.
    Step 10: Save all outputs.

    Run from terminal:

        py src/evaluate_all_models.py baseline
        py src/evaluate_all_models.py improved
        py src/evaluate_all_models.py transfer
    """

    if len(sys.argv) < 2:
        print("Usage: py src/evaluate_all_models.py [baseline|improved|transfer]")
        sys.exit(1)

    model_type = sys.argv[1].lower()

    if model_type not in ["baseline", "improved", "transfer"]:
        print("model_type must be one of: baseline, improved, transfer")
        sys.exit(1)

    # Step 2: Load CIFAR-10 test data.
    # We only need x_test and y_test for evaluation.
    _, _, x_test, y_test, class_names = load_cifar10()

    # Step 1: Load the chosen model.
    model, config = load_trained_model(model_type)
    prefix = config["display_name"]

    # Step 3: Evaluate test accuracy/loss.
    # Step 4: Predict classes.
    _, _, y_true, y_pred, y_prob = evaluate_model(model, x_test, y_test, model_type)

    # Step 5: Create confusion matrix.
    plot_confusion_matrix(y_true, y_pred, class_names, prefix)

    # Step 6: Create classification report.
    print_classification_report_file(y_true, y_pred, class_names, prefix)

    # Extra: Print top confused class pairs for report discussion.
    print_confusion_analysis(y_true, y_pred, class_names, top_n=5)

    # Step 7: Show correct predictions.
    show_prediction_examples(
        x_test,
        y_true,
        y_pred,
        y_prob,
        class_names,
        prefix=prefix,
        correct=True,
        num_examples=5
    )

    # Step 8: Show incorrect predictions.
    show_prediction_examples(
        x_test,
        y_true,
        y_pred,
        y_prob,
        class_names,
        prefix=prefix,
        correct=False,
        num_examples=5
    )

    # Step 9: Try Grad-CAM.
    # Step 10: Save Grad-CAM output.
    display_gradcam(
        x_test,
        y_true,
        y_pred,
        model,
        class_names,
        model_type=model_type,
        prefix=prefix,
        num_examples=5
    )

    print(f"\nEvaluation completed for: {model_type}")
    print("Check the outputs folder for saved files.")


if __name__ == "__main__":
    main()