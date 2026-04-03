import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.data_loader import load_cifar10
from src.model import build_baseline_cnn

x_train, y_train, x_test, y_test, class_names = load_cifar10()

model = build_baseline_cnn()

history = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.2
)

test_loss, test_acc = model.evaluate(x_test, y_test)
print("Baseline Test Accuracy:", test_acc)