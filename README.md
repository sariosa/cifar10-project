# CIFAR-10 & CIFAR-100 Image Classification

A CNN-based image classification project covering CIFAR-10 and CIFAR-100. Includes baseline training, data augmentation, transfer learning with ResNet50, and an improved CNN with BatchNormalization and adaptive callbacks. The CIFAR-100 section applies all lessons learned from CIFAR-10 from the start.

---

## Setup

**Requirements:** Python 3.9

### 1. Clone the repository
```bash
git clone https://github.com/sehajreetkaur/cifar10-project.git
cd cifar10-project
```

### 2. Create and activate a virtual environment
```bash
python3.9 -m venv venv

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

---

## CIFAR-100

The `cifar100/` folder applies every lesson from CIFAR-10 to the harder CIFAR-100 dataset (100 classes, only 500 images per class).

### Key improvements over CIFAR-10

| | CIFAR-10 | CIFAR-100 |
|---|---|---|
| Conv blocks | 3 | 4 (128→256→512→512) |
| Pooling before classifier | Flatten | GlobalAveragePooling |
| Dropout | 0.4 | 0.5 |
| L2 regularization | No | Yes |
| Augmentation strength | Basic | Stronger (+ shear) |
| Callbacks in transfer learning | No | Yes |
| Fine-tune layers | All base | Top 50 only |
| Early stop patience | 7 | 10 |

### How to Run (CIFAR-100)

Run all commands from the repo root with the venv activated.

#### CNN Training
```bash
python cifar100/src/train.py
```
Trains with augmentation and all callbacks from epoch 1. Saves best model to `cifar100/outputs/cnn_cifar100.keras` and training curves to `cifar100/outputs/training_curves.png`.

To view TensorBoard logs:
```bash
tensorboard --logdir=cifar100/logs
```

#### Transfer Learning (EfficientNetB3)
```bash
python cifar100/src/transfer_learning.py
```
Downloads EfficientNetB3 weights on first run (~44MB). Resizes images to 224×224 on-the-fly via a tf.data pipeline (avoids RAM overflow). Trains the custom head for up to 15 epochs (base frozen), then fine-tunes the top 50 EfficientNetB3 layers for up to 15 more epochs at a low learning rate. Uses label smoothing (0.1) and cosine decay LR schedule throughout. Achieved **77.46% test accuracy**. Saves model to `cifar100/outputs/efficientnetb3_cifar100.keras`.

#### Predict on an image
```bash
python cifar100/src/predict.py --image path/to/image.jpg

# Use transfer learning model instead
python cifar100/src/predict.py --image path/to/image.jpg --model cifar100/outputs/resnet50_cifar100.keras
```
Prints top-5 predictions with confidence scores and saves a bar chart to `cifar100/outputs/prediction.png`.

### What Each File Does (CIFAR-100)

#### `cifar100/src/data_loader.py`
Loads CIFAR-100 fine labels (100 classes), normalises pixels to [0,1], and one-hot encodes labels. Returns train/test splits and all 100 class names.

#### `cifar100/src/model_cnn.py`
Deeper CNN with 4 conv blocks, BatchNorm, LeakyReLU, GlobalAveragePooling, L2 regularization, and Dropout(0.5). Outputs 100-class softmax.

#### `cifar100/src/augmentation.py`
Stronger augmentation than CIFAR-10: rotation 15°, shifts 10%, shear 10%, zoom 15%, horizontal flip. Applied from the first training epoch.

#### `cifar100/src/train.py`
Full training pipeline with augmentation + ReduceLROnPlateau + EarlyStopping + ModelCheckpoint baked in from day 1. Saves best model, training curves, and a results JSON to `cifar100/outputs/`.

#### `cifar100/src/transfer_learning.py`
- Loads EfficientNetB3 pretrained on ImageNet
- Resizes CIFAR-100 images from 32×32 to 224×224 on-the-fly (tf.data pipeline, no RAM spike)
- Phase 1: trains custom head (GlobalAveragePooling → BatchNorm → Dense 512 → Dropout → Dense 100) with base frozen, cosine decay LR from 1e-3
- Phase 2: fine-tunes top 50 EfficientNetB3 layers at cosine decay LR from 1e-5
- Label smoothing (0.1) applied in both phases
- Both phases use EarlyStopping and ModelCheckpoint
- Achieved **77.46% test accuracy** on CIFAR-100
- Saves model and training curves to `cifar100/outputs/`

#### `cifar100/src/predict.py`
Loads any saved CIFAR-100 model, runs inference on a single image, and shows top-5 class predictions with a confidence bar chart.
