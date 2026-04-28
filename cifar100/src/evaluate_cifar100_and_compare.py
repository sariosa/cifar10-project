import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.applications.efficientnet import preprocess_input


# Project paths

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

CIFAR10_OUTPUT_DIR = os.path.join(project_root, "outputs")
CIFAR100_OUTPUT_DIR = os.path.join(project_root, "cifar100", "outputs")

os.makedirs(CIFAR100_OUTPUT_DIR, exist_ok=True)


# CIFAR-100 settings

CIFAR100_MODEL_PATH = os.path.join(
    CIFAR100_OUTPUT_DIR,
    "efficientnetb3_cifar100_finetune.keras"
)

CIFAR100_CLASSES = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
    "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
    "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
    "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster",
    "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
    "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain",
    "mouse", "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree",
    "pear", "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine",
    "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea",
    "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider",
    "squirrel", "streetcar", "sunflower", "sweet_pepper", "table", "tank",
    "telephone", "television", "tiger", "tractor", "train", "trout", "tulip",
    "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm"
]

AUTOTUNE = tf.data.AUTOTUNE


# Helper: read JSON result files

def read_accuracy_from_json(path):
    """
    Read test accuracy from a JSON file.

    Different scripts may save the accuracy under slightly different keys,
    so this function checks common names.
    """
    if not os.path.exists(path):
        return None

    with open(path, "r") as f:
        data = json.load(f)

    possible_keys = [
        "test_accuracy",
        "transfer_learning_test_accuracy",
        "accuracy"
    ]

    for key in possible_keys:
        if key in data:
            return float(data[key])

    return None


def read_loss_from_json(path):
    """
    Read test loss from a JSON file if available.
    """
    if not os.path.exists(path):
        return None

    with open(path, "r") as f:
        data = json.load(f)

    possible_keys = [
        "test_loss",
        "transfer_learning_test_loss",
        "loss"
    ]

    for key in possible_keys:
        if key in data:
            return float(data[key])

    return None


# Step 1: Load CIFAR-100 fine-tuned model

def load_cifar100_model():
    """
    Load the final fine-tuned EfficientNetB3 CIFAR-100 model.
    """
    if not os.path.exists(CIFAR100_MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at:\n{CIFAR100_MODEL_PATH}\n\n"
            "Make sure this file exists inside cifar100/outputs/:\n"
            "efficientnetb3_cifar100_finetune.keras"
        )

    print(f"Loading CIFAR-100 model from: {CIFAR100_MODEL_PATH}")
    model = tf.keras.models.load_model(CIFAR100_MODEL_PATH)

    print(f"Model input shape: {model.input_shape}")
    return model


# Step 2: Load CIFAR-100 test data

def load_cifar100_test_data():
    """
    Load CIFAR-100 test data from Keras.

    Returns:
    - x_test: test images
    - y_test_labels: integer labels, used for reports/confusion matrix
    - y_test_onehot: one-hot labels, used for model.evaluate()
    """
    print("\nLoading CIFAR-100 test data...")

    (_, _), (x_test, y_test) = cifar100.load_data(label_mode="fine")

    y_test_labels = y_test.flatten()
    y_test_onehot = tf.keras.utils.to_categorical(y_test_labels, num_classes=100)

    print(f"x_test shape: {x_test.shape}")
    print(f"y_test_labels shape: {y_test_labels.shape}")
    print(f"y_test_onehot shape: {y_test_onehot.shape}")

    return x_test, y_test_labels, y_test_onehot


# Step 3: Preprocess CIFAR-100 images for EfficientNetB3

def preprocess_cifar100_image(image, label, target_size):
    """
    Resize CIFAR-100 image and apply EfficientNet preprocessing.
    """
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, target_size)
    image = preprocess_input(image)

    return image, label


def create_cifar100_test_dataset(x_test, y_test_onehot, model, batch_size=32):
    """
    Create memory-safe TensorFlow dataset for CIFAR-100 evaluation.

    This avoids resizing all 10,000 test images at once.
    Uses one-hot labels because the model was compiled with categorical_crossentropy.
    """
    input_shape = model.input_shape

    height = input_shape[1]
    width = input_shape[2]

    if height is None or width is None:
        height, width = 300, 300

    target_size = (height, width)

    print(f"Resizing CIFAR-100 test images to: {target_size}")

    dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test_onehot))
    dataset = dataset.map(
        lambda img, label: preprocess_cifar100_image(img, label, target_size),
        num_parallel_calls=AUTOTUNE
    )
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTOTUNE)

    return dataset


# Step 4: Evaluate CIFAR-100 model

def evaluate_cifar100_model(model, test_dataset):
    """
    Evaluate CIFAR-100 model and generate predictions.
    """
    print("\nEvaluating CIFAR-100 EfficientNetB3 model...")

    test_loss, test_acc = model.evaluate(test_dataset, verbose=1)

    print(f"\nCIFAR-100 Test Accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)")
    print(f"CIFAR-100 Test Loss:     {test_loss:.4f}")

    print("\nPredicting CIFAR-100 classes...")
    y_prob = model.predict(test_dataset, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)

    return test_loss, test_acc, y_pred, y_prob


# Step 5: Save CIFAR-100 evaluation outputs

def save_cifar100_results_json(test_acc, test_loss):
    """
    Save CIFAR-100 evaluation accuracy/loss.
    """
    results = {
        "model": "EfficientNetB3 fine-tuned",
        "dataset": "CIFAR-100",
        "test_accuracy": round(float(test_acc), 4),
        "test_loss": round(float(test_loss), 4)
    }

    save_path = os.path.join(CIFAR100_OUTPUT_DIR, "cifar100_evaluation_results.json")

    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"CIFAR-100 evaluation results saved to: {save_path}")


def save_cifar100_classification_report(y_true, y_pred):
    """
    Save CIFAR-100 classification report.
    """
    report = classification_report(
        y_true,
        y_pred,
        target_names=CIFAR100_CLASSES,
        digits=4
    )

    print("\nCIFAR-100 Classification Report:\n")
    print(report)

    save_path = os.path.join(CIFAR100_OUTPUT_DIR, "cifar100_classification_report.txt")

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"CIFAR-100 classification report saved to: {save_path}")


def save_cifar100_confusion_matrix(y_true, y_pred):
    """
    Save CIFAR-100 confusion matrix.

    CIFAR-100 has 100 classes, so this image will be large.
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(24, 24))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CIFAR100_CLASSES
    )

    disp.plot(
        ax=ax,
        xticks_rotation=90,
        colorbar=False,
        cmap="Blues",
        values_format="d"
    )

    plt.title("Confusion Matrix - CIFAR-100 EfficientNetB3 Fine-Tuned")
    plt.tight_layout()

    save_path = os.path.join(CIFAR100_OUTPUT_DIR, "cifar100_confusion_matrix.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()

    print(f"CIFAR-100 confusion matrix saved to: {save_path}")


def save_cifar100_prediction_examples(x_test, y_true, y_pred, y_prob, correct=True, num_examples=8):
    """
    Save CIFAR-100 correct or incorrect prediction examples.
    """
    if correct:
        indices = np.where(y_true == y_pred)[0]
        title = "CIFAR-100 Correct Predictions"
        save_name = "cifar100_correct_predictions.png"
    else:
        indices = np.where(y_true != y_pred)[0]
        title = "CIFAR-100 Incorrect Predictions"
        save_name = "cifar100_incorrect_predictions.png"

    if len(indices) == 0:
        print(f"No {title.lower()} found.")
        return

    chosen = indices[:num_examples]

    plt.figure(figsize=(18, 4))

    for i, idx in enumerate(chosen):
        plt.subplot(1, num_examples, i + 1)
        plt.imshow(x_test[idx])

        pred_class = CIFAR100_CLASSES[y_pred[idx]]
        true_class = CIFAR100_CLASSES[y_true[idx]]
        confidence = y_prob[idx][y_pred[idx]] * 100

        plt.title(
            f"Pred: {pred_class}\nTrue: {true_class}\n{confidence:.1f}%",
            fontsize=8
        )

        plt.axis("off")

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    save_path = os.path.join(CIFAR100_OUTPUT_DIR, save_name)
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()

    print(f"{title} saved to: {save_path}")


def print_cifar100_top_confusions(y_true, y_pred, top_n=10):
    """
    Print top CIFAR-100 class confusions for report discussion.
    """
    cm = confusion_matrix(y_true, y_pred)
    confusion_pairs = []

    for i in range(len(CIFAR100_CLASSES)):
        for j in range(len(CIFAR100_CLASSES)):
            if i != j and cm[i, j] > 0:
                confusion_pairs.append((cm[i, j], CIFAR100_CLASSES[i], CIFAR100_CLASSES[j]))

    confusion_pairs.sort(reverse=True, key=lambda x: x[0])

    print(f"\nTop {top_n} most common CIFAR-100 confusions:")

    for count, true_class, pred_class in confusion_pairs[:top_n]:
        print(f"  True {true_class:>15} -> Pred {pred_class:<15} : {count}")


# Step 6: Build side-by-side CIFAR-10 vs CIFAR-100 comparison

def collect_project_results(cifar100_eval_acc=None, cifar100_eval_loss=None):
    """
    Collect available CIFAR-10 and CIFAR-100 results from JSON files.

    This creates one comparison table across both datasets.
    Missing files are skipped automatically.
    """
    results = []

    # CIFAR-10 results

    c10_baseline_path = os.path.join(CIFAR10_OUTPUT_DIR, "baseline_results.json")
    c10_improved_v1_path = os.path.join(CIFAR10_OUTPUT_DIR, "improved_results.json")
    c10_improved_v2_path = os.path.join(CIFAR10_OUTPUT_DIR, "improved_v2_results.json")
    c10_improved_v3_path = os.path.join(CIFAR10_OUTPUT_DIR, "improved_v3_results.json")
    c10_transfer_path = os.path.join(CIFAR10_OUTPUT_DIR, "transfer_learning_results.json")

    c10_baseline_acc = read_accuracy_from_json(c10_baseline_path)
    c10_baseline_loss = read_loss_from_json(c10_baseline_path)

    if c10_baseline_acc is not None:
        results.append({
            "dataset": "CIFAR-10",
            "stage": "Baseline",
            "model": "Baseline CNN",
            "accuracy": c10_baseline_acc,
            "loss": c10_baseline_loss,
            "evidence": "outputs/baseline_results.json"
        })

    c10_improved_v1_acc = read_accuracy_from_json(c10_improved_v1_path)
    c10_improved_v1_loss = read_loss_from_json(c10_improved_v1_path)

    if c10_improved_v1_acc is not None:
        results.append({
            "dataset": "CIFAR-10",
            "stage": "Improved",
            "model": "Improved CNN v1",
            "accuracy": c10_improved_v1_acc,
            "loss": c10_improved_v1_loss,
            "evidence": "outputs/improved_results.json"
        })

    c10_improved_v2_acc = read_accuracy_from_json(c10_improved_v2_path)
    c10_improved_v2_loss = read_loss_from_json(c10_improved_v2_path)

    if c10_improved_v2_acc is not None:
        results.append({
            "dataset": "CIFAR-10",
            "stage": "Improved",
            "model": "Improved CNN v2",
            "accuracy": c10_improved_v2_acc,
            "loss": c10_improved_v2_loss,
            "evidence": "outputs/improved_v2_results.json"
        })

    c10_improved_v3_acc = read_accuracy_from_json(c10_improved_v3_path)
    c10_improved_v3_loss = read_loss_from_json(c10_improved_v3_path)

    if c10_improved_v3_acc is not None:
        results.append({
            "dataset": "CIFAR-10",
            "stage": "Improved",
            "model": "Improved CNN v3",
            "accuracy": c10_improved_v3_acc,
            "loss": c10_improved_v3_loss,
            "evidence": "outputs/improved_v3_results.json"
        })

    c10_transfer_acc = read_accuracy_from_json(c10_transfer_path)
    c10_transfer_loss = read_loss_from_json(c10_transfer_path)

    if c10_transfer_acc is not None:
        results.append({
            "dataset": "CIFAR-10",
            "stage": "Transfer Learning",
            "model": "ResNet50 transfer learning",
            "accuracy": c10_transfer_acc,
            "loss": c10_transfer_loss,
            "evidence": "outputs/transfer_learning_results.json"
        })

    # CIFAR-100 results

    c100_cnn_path = os.path.join(CIFAR100_OUTPUT_DIR, "cnn_results.json")
    c100_transfer_path = os.path.join(CIFAR100_OUTPUT_DIR, "transfer_results.json")

    c100_cnn_acc = read_accuracy_from_json(c100_cnn_path)
    c100_cnn_loss = read_loss_from_json(c100_cnn_path)

    if c100_cnn_acc is not None:
        results.append({
            "dataset": "CIFAR-100",
            "stage": "Baseline/From Scratch",
            "model": "CIFAR-100 CNN",
            "accuracy": c100_cnn_acc,
            "loss": c100_cnn_loss,
            "evidence": "cifar100/outputs/cnn_results.json"
        })

    c100_transfer_acc = read_accuracy_from_json(c100_transfer_path)
    c100_transfer_loss = read_loss_from_json(c100_transfer_path)

    # Prefer the evaluation result from the model we just evaluated.
    if cifar100_eval_acc is not None:
        c100_transfer_acc = cifar100_eval_acc
        c100_transfer_loss = cifar100_eval_loss

    if c100_transfer_acc is not None:
        results.append({
            "dataset": "CIFAR-100",
            "stage": "Transfer Learning",
            "model": "EfficientNetB3 fine-tuned",
            "accuracy": c100_transfer_acc,
            "loss": c100_transfer_loss,
            "evidence": "cifar100/outputs/transfer_results.json / cifar100_evaluation_results.json"
        })

    return results


def save_comparison_table(results):
    """
    Save side-by-side comparison table as CSV and TXT.
    """
    csv_path = os.path.join(project_root, "project_model_comparison.csv")
    txt_path = os.path.join(project_root, "project_model_comparison.txt")

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("Dataset,Stage,Model,Accuracy,Accuracy (%),Loss,Evidence\n")

        for row in results:
            accuracy = row["accuracy"]
            loss = row["loss"]

            f.write(
                f"{row['dataset']},"
                f"{row['stage']},"
                f"{row['model']},"
                f"{accuracy:.4f},"
                f"{accuracy * 100:.2f},"
                f"{'' if loss is None else round(float(loss), 4)},"
                f"{row['evidence']}\n"
            )

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("PROJECT MODEL COMPARISON\n")
        f.write("=" * 80 + "\n\n")

        for row in results:
            accuracy = row["accuracy"]
            loss = row["loss"]

            f.write(f"Dataset   : {row['dataset']}\n")
            f.write(f"Stage     : {row['stage']}\n")
            f.write(f"Model     : {row['model']}\n")
            f.write(f"Accuracy  : {accuracy:.4f} ({accuracy * 100:.2f}%)\n")

            if loss is not None:
                f.write(f"Loss      : {float(loss):.4f}\n")

            f.write(f"Evidence  : {row['evidence']}\n")
            f.write("-" * 80 + "\n")

    print(f"\nComparison CSV saved to: {csv_path}")
    print(f"Comparison TXT saved to: {txt_path}")


def save_comparison_chart(results):
    """
    Save a side-by-side comparison chart of CIFAR-10 and CIFAR-100 models.
    """
    labels = [f"{row['dataset']}\n{row['model']}" for row in results]
    accuracies = [row["accuracy"] * 100 for row in results]

    plt.figure(figsize=(14, 6))
    bars = plt.bar(labels, accuracies)

    plt.ylabel("Test Accuracy (%)")
    plt.title("CIFAR-10 vs CIFAR-100 Model Performance Comparison")
    plt.ylim(0, 100)
    plt.xticks(rotation=30, ha="right")

    for bar, acc in zip(bars, accuracies):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{acc:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9
        )

    plt.tight_layout()

    save_path = os.path.join(project_root, "project_model_comparison.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()

    print(f"Comparison chart saved to: {save_path}")


def print_comparison_summary(results):
    """
    Print comparison table in the terminal.
    """
    print("\n" + "=" * 90)
    print("SIDE-BY-SIDE MODEL COMPARISON")
    print("=" * 90)

    for row in results:
        loss_text = "" if row["loss"] is None else f" | Loss: {float(row['loss']):.4f}"

        print(
            f"{row['dataset']:<10} | "
            f"{row['stage']:<20} | "
            f"{row['model']:<30} | "
            f"Accuracy: {row['accuracy'] * 100:>6.2f}%"
            f"{loss_text}"
        )

    print("=" * 90)


# Main

def main():
    """
    Full pipeline:

    CIFAR-100 evaluation:
      1. Load EfficientNetB3 fine-tuned model
      2. Load CIFAR-100 test data
      3. Preprocess test images
      4. Evaluate test accuracy/loss
      5. Save classification report, confusion matrix, and prediction examples

    Project comparison:
      6. Read CIFAR-10 and CIFAR-100 result JSON files
      7. Create side-by-side comparison table and chart
    """
    model = load_cifar100_model()

    x_test, y_test_labels, y_test_onehot = load_cifar100_test_data()

    test_dataset = create_cifar100_test_dataset(
        x_test,
        y_test_onehot,
        model,
        batch_size=32
    )

    test_loss, test_acc, y_pred, y_prob = evaluate_cifar100_model(
        model,
        test_dataset
    )

    save_cifar100_results_json(test_acc, test_loss)

    save_cifar100_classification_report(y_test_labels, y_pred)

    save_cifar100_confusion_matrix(y_test_labels, y_pred)

    print_cifar100_top_confusions(y_test_labels, y_pred, top_n=10)

    save_cifar100_prediction_examples(
        x_test,
        y_test_labels,
        y_pred,
        y_prob,
        correct=True,
        num_examples=8
    )

    save_cifar100_prediction_examples(
        x_test,
        y_test_labels,
        y_pred,
        y_prob,
        correct=False,
        num_examples=8
    )

    comparison_results = collect_project_results(
        cifar100_eval_acc=test_acc,
        cifar100_eval_loss=test_loss
    )

    print_comparison_summary(comparison_results)

    save_comparison_table(comparison_results)

    save_comparison_chart(comparison_results)

    print("\nDone.")
    print("CIFAR-100 evaluation outputs are in: cifar100/outputs/")
    print("Project comparison files are in the project root:")
    print("- project_model_comparison.csv")
    print("- project_model_comparison.txt")
    print("- project_model_comparison.png")


if __name__ == "__main__":
    main()