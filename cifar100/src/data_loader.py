"""
data_loader.py

Loads and preprocesses the CIFAR-100 dataset.

CIFAR-100 vs CIFAR-10:
- 100 fine-grained classes instead of 10
- Same 60,000 images total — only 500 per class instead of 5,000
- This makes it a much harder problem and more prone to overfitting
- Augmentation and regularization matter even more here

How to use:
    from cifar100.src.data_loader import load_cifar100
    x_train, y_train, x_test, y_test, class_names = load_cifar100()
"""

from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical

# Standard CIFAR-100 fine class names (100 classes)
CIFAR100_CLASSES = [
    "apple", "aquarium_fish", "baby", "bear", "beaver",
    "bed", "bee", "beetle", "bicycle", "bottle",
    "bowl", "boy", "bridge", "bus", "butterfly",
    "camel", "can", "castle", "caterpillar", "cattle",
    "chair", "chimpanzee", "clock", "cloud", "cockroach",
    "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox",
    "girl", "hamster", "house", "kangaroo", "keyboard",
    "lamp", "lawn_mower", "leopard", "lion", "lizard",
    "lobster", "man", "maple_tree", "motorcycle", "mountain",
    "mouse", "mushroom", "oak_tree", "orange", "orchid",
    "otter", "palm_tree", "pear", "pickup_truck", "pine_tree",
    "plain", "plate", "poppy", "porcupine", "possum",
    "rabbit", "raccoon", "ray", "road", "rocket",
    "rose", "sea", "seal", "shark", "shrew",
    "skunk", "skyscraper", "snail", "snake", "spider",
    "squirrel", "streetcar", "sunflower", "sweet_pepper", "table",
    "tank", "telephone", "television", "tiger", "tractor",
    "train", "trout", "tulip", "turtle", "wardrobe",
    "whale", "willow_tree", "wolf", "woman", "worm",
]


def load_cifar100():
    """
    Load and preprocess CIFAR-100.

    Steps:
    - Normalise pixel values to [0, 1]
    - One-hot encode labels across 100 classes

    Returns:
        x_train:     (50000, 32, 32, 3) float32
        y_train:     (50000, 100) one-hot
        x_test:      (10000, 32, 32, 3) float32
        y_test:      (10000, 100) one-hot
        class_names: list of 100 class name strings
    """
    (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode="fine")

    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32")  / 255.0

    y_train = to_categorical(y_train, 100)
    y_test  = to_categorical(y_test,  100)

    return x_train, y_train, x_test, y_test, CIFAR100_CLASSES
