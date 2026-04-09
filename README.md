# CIFAR-10 Image Classification

A CNN-based image classification project on the CIFAR-10 dataset. Covers baseline training, data augmentation, transfer learning with ResNet50, and an improved CNN with BatchNormalization and adaptive training callbacks.

---

## Setup

**Requirements:** Python 3.12 (not Python 3.9)

### 1. Clone the repository
```bash
git clone https://github.com/sehajreetkaur/cifar10-project.git
cd cifar10-project
```

### 2. Create and activate a virtual environment
```bash
python3.12 -m venv venv

# Mac/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## How to Run

### Baseline CNN
```bash
python src/train_baseline.py
```
Trains for 10 epochs, prints test accuracy, saves model to `outputs/CNN.keras`, and saves `outputs/baseline_results.json` for use by the improved model comparison.

### Data Augmentation
```bash
python src/augmentation.py
```
Shows augmented image previews, trains both runs, plots and saves comparison curves.

To view TensorBoard logs after training:
```bash
tensorboard --logdir=logs/augmentation
```

### Transfer Learning
```bash
python src/transfer_learning.py
```
Downloads ResNet50 weights (first run only), trains in two phases, prints accuracy comparison, saves model and plots to `outputs/`.

### Improved CNN
```bash
python src/train_improved.py
```
Trains the improved architecture with augmented data and adaptive callbacks. Run `src/train_baseline.py` first if `outputs/baseline_results.json` does not exist — it is needed to print the comparison table. Saves the best model to `outputs/CNN_improved.keras` and training curves to `outputs/`.

To view TensorBoard logs after training:
```bash
tensorboard --logdir=logs/improved
```

---

## What Each File Does

### `src/data_loader.py`
Loads CIFAR-10 from Keras, normalises pixel values to [0,1], and one-hot encodes the labels. Returns train/test splits and class names.

### `src/model_CNN.py`
Defines the baseline CNN architecture:
- 3 × Conv2D layers (32 → 64 → 64 filters)
- MaxPooling after each of the first two
- Dense(64) → Dense(10, softmax)
- Compiled with Adam + categorical crossentropy

### `src/train_baseline.py`
Trains the baseline CNN on CIFAR-10 for 10 epochs. Evaluates on the test set, saves the model to `outputs/CNN.keras`, and writes `outputs/baseline_results.json` so the improved training script can read the real baseline accuracy for comparison.

### `src/augmentation.py`
- Applies augmentation (rotation, shifts, zoom, horizontal flip)
- Previews original vs augmented images
- Trains the CNN without and with augmentation for comparison
- Plots training curves for both runs side by side
- Saves comparison plot to `outputs/augmentation_comparison.png`
- Logs to TensorBoard at `logs/augmentation/`

### `src/transfer_learning.py`
- Loads ResNet50 pretrained on ImageNet (no top layer)
- Resizes CIFAR-10 images from 32×32 to 224×224
- Adds a custom head: GlobalAveragePooling → Dense(256) → Dropout → Dense(10)
- Phase 1: trains the head only (base frozen, 10 epochs)
- Phase 2: fine-tunes the full model at a low learning rate (5 epochs)
- Compares accuracy vs baseline CNN
- Saves model to `outputs/resnet50_cifar10.keras`
- Saves architecture diagram and training curves to `outputs/`

### `src/model_improved_cnn.py`
Defines the improved CNN architecture:
- 3 × Conv2D layers (64 → 128 → 256 filters) with `padding='same'`
- BatchNormalization after each Conv2D
- LeakyReLU(0.1) activations throughout (prevents dead neurons)
- Dense(256) → Dropout(0.4) → Dense(10, softmax)
- Compiled with Adam + categorical crossentropy

### `src/train_improved.py`
Trains the improved CNN on CIFAR-10 with augmented data and three callbacks:
- `ReduceLROnPlateau` — halves the learning rate when val_accuracy stalls for 3 epochs
- `EarlyStopping` — stops training when val_loss stops improving (patience 7), restores best weights
- `ModelCheckpoint` — saves the best model to `outputs/CNN_improved.keras`

Reads `outputs/baseline_results.json` to print a live accuracy comparison at the end. Saves final model, training curves, and results JSON to `outputs/`.