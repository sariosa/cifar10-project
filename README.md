# CIFAR-10 & CIFAR-100 Image Classification

CNN-based image classification project covering both CIFAR-10 and CIFAR-100, with experiments ranging from a simple baseline CNN to transfer learning and a stronger residual CNN with progressive regularization. The repository is organized as a reproducible coursework-style project: training scripts, saved result files, prediction utilities, and lightweight tests are all included.

The main outcome is a clear accuracy progression on CIFAR-10 from a 70.0% baseline to 92.14% test accuracy, then application of the same training lessons to CIFAR-100, where EfficientNetB3 transfer learning reaches 77.55% test accuracy.

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Layout](#repository-layout)
- [Results](#results)
- [Setup](#setup)
- [How to Run](#how-to-run)
- [Methodology Highlights](#methodology-highlights)
- [Key Files](#key-files)
- [Reproducibility Notes](#reproducibility-notes)
- [References](#references)

## Project Overview

- **Datasets:** CIFAR-10 and CIFAR-100, both loaded from `tf.keras.datasets`
- **CIFAR-10 scope:** baseline CNN, augmentation study, ResNet50 transfer learning, and improved residual CNN
- **CIFAR-100 scope:** stronger CNN pipeline from the start, EfficientNetB3 transfer learning, and prediction utilities
- **Focus:** architecture design, regularization, augmentation, learning-rate scheduling, and reproducible result tracking

Dataset facts:

- **CIFAR-10:** 60,000 color images, 10 classes, 32x32 resolution
- **CIFAR-100:** 60,000 color images, 100 classes, 32x32 resolution
- **CIFAR-100 split:** 500 training images and 100 test images per class

## Repository Layout

```text
.
├── src/
│   ├── augmentation.py
│   ├── data_loader.py
│   ├── model_CNN.py
│   ├── model_improved_cnn.py
│   ├── predict.py
│   ├── train_baseline.py
│   ├── train_improved.py
│   └── transfer_learning.py
├── cifar100/
│   └── src/
│       ├── augmentation.py
│       ├── data_loader.py
│       ├── model_cnn.py
│       ├── predict.py
│       ├── train.py
│       └── transfer_learning.py
├── outputs/
├── cifar100/outputs/
├── test/
├── requirements.txt
├── how_to.txt
└── README.md
```

## Results

### CIFAR-10

The table below uses the saved result files currently present in `outputs/`. All values are **test accuracy**.

| Model | Test Accuracy | Evidence | Main change |
|---|---:|---|---|
| Baseline CNN | 70.00% | `outputs/baseline_results.json` | 3-block plain CNN |
| Improved CNN v1 | 86.18% | `outputs/improved_results.json` | BatchNorm, LeakyReLU, stronger head |
| Improved CNN v2 | 91.53% | `outputs/improved_v2_results.json` | longer schedule, dropout, label smoothing |
| Improved CNN v3 | **92.14%** | `outputs/improved_v3_results.json` | residual blocks, conv weight decay, stronger augmentation, cutout |

### CIFAR-100

The strongest committed CIFAR-100 result currently saved in the repository is the transfer-learning run below.

| Model | Test Accuracy | Evidence | Main change |
|---|---:|---|---|
| EfficientNetB3 transfer learning | **77.55%** | `cifar100/outputs/transfer_results.json` | pretrained backbone, cosine decay, label smoothing, fine-tuning top 50 layers |

Note:

- `cifar100/src/train.py` provides a from-scratch CNN training pipeline, but its output JSON is not currently committed in `cifar100/outputs/`
- Validation strategy differs slightly across scripts, so the fairest headline comparisons are the saved test accuracies above

## Setup

**Recommended Python version:** 3.12

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

## How to Run

Run commands from the repository root with the virtual environment activated.

### CIFAR-10

#### Baseline CNN

```bash
python src/train_baseline.py
```

Trains the baseline model for 10 epochs, saves `outputs/CNN.keras`, writes `outputs/baseline_results.json`, and saves `outputs/baseline_training_curves.png`.

#### Augmentation demo

```bash
python src/augmentation.py
```

Shows augmentation previews and runs a baseline-vs-augmented comparison experiment. Outputs include `outputs/augmentation_comparison.png` and TensorBoard logs in `logs/augmentation/`.

View logs with:

```bash
tensorboard --logdir=logs/augmentation
```

#### Improved CNN v3

```bash
python src/train_improved.py
```

Trains the current best CIFAR-10 model. If `outputs/baseline_results.json` is missing, run `src/train_baseline.py` first so the script can print the comparison table.

Outputs:

- `outputs/CNN_improved_v3.keras` for the best checkpoint
- `outputs/CNN_improved_v3_final.keras` for the final saved model state
- `outputs/improved_v3_results.json`
- `outputs/improved_v3_training_curves.png`
- TensorBoard logs in `logs/improved_v3/`

View logs with:

```bash
tensorboard --logdir=logs/improved_v3
```

#### Transfer learning with ResNet50

```bash
python src/transfer_learning.py
```

Runs a two-phase CIFAR-10 transfer-learning experiment with ResNet50 and saves the trained model plus plots to `outputs/`.

#### Predict on a CIFAR-10 image

```bash
python src/predict.py --model outputs/CNN_improved_v3.keras --image path/to/image.png
```

If `--image` is omitted, the script uses a random sample from the CIFAR-10 test set.

### CIFAR-100

#### CNN training

```bash
python cifar100/src/train.py
```

Trains the CIFAR-100 CNN with augmentation, cosine decay, label smoothing, checkpointing, and early stopping. Expected outputs include `cifar100/outputs/cnn_cifar100.keras`, `cifar100/outputs/cnn_results.json`, and `cifar100/outputs/training_curves.png`.

View logs with:

```bash
tensorboard --logdir=cifar100/logs
```

#### Transfer learning with EfficientNetB3

```bash
python cifar100/src/transfer_learning.py
```

Runs two-phase transfer learning on CIFAR-100 and saves:

- `cifar100/outputs/efficientnetb3_cifar100.keras`
- `cifar100/outputs/efficientnetb3_cifar100_head.keras`
- `cifar100/outputs/efficientnetb3_cifar100_finetune.keras`
- `cifar100/outputs/transfer_results.json`
- `cifar100/outputs/transfer_learning_curves.png`

#### Predict on an image

```bash
python cifar100/src/predict.py --image path/to/image.jpg
```

To use the transfer-learning model instead of the default CNN checkpoint:

```bash
python cifar100/src/predict.py --image path/to/image.jpg --model cifar100/outputs/efficientnetb3_cifar100.keras
```

## Methodology Highlights

### CIFAR-10 improved model

- Residual architecture with stages `64 -> 128 -> 256 -> 512`
- Batch normalization and `LeakyReLU(0.1)` throughout
- Learnable downsampling via stride-2 convolutions instead of max pooling
- Global average pooling before the classifier head
- Conv-kernel weight decay (`5e-5`) and dense-layer weight decay (`1e-4`)
- Dropout (`0.5`), label smoothing (`0.1`), cosine decay learning rate schedule
- Stronger augmentation with horizontal flips, translations, zoom, and cutout

### CIFAR-100 pipeline

- Stronger augmentation than CIFAR-10 from the start
- Dedicated CIFAR-100 CNN with four convolutional stages and global average pooling
- EfficientNetB3 transfer learning with on-the-fly resizing through `tf.data`
- Fine-tuning restricted to the top 50 EfficientNetB3 layers
- Label smoothing and cosine decay in both phases of transfer learning

## Key Files

| Path | Purpose |
|---|---|
| `src/train_baseline.py` | Baseline CIFAR-10 training and result logging |
| `src/augmentation.py` | CIFAR-10 augmentation preview and comparison experiment |
| `src/model_improved_cnn.py` | Residual CIFAR-10 architecture definition |
| `src/train_improved.py` | Main CIFAR-10 improved training pipeline |
| `src/transfer_learning.py` | CIFAR-10 ResNet50 transfer-learning experiment |
| `src/predict.py` | CIFAR-10 single-image inference |
| `cifar100/src/train.py` | CIFAR-100 CNN training pipeline |
| `cifar100/src/transfer_learning.py` | CIFAR-100 EfficientNetB3 transfer learning |
| `cifar100/src/predict.py` | CIFAR-100 single-image inference |
| `test/` | Lightweight smoke tests for loaders, augmentation, CNNs, and transfer learning |

## Reproducibility Notes

- The project uses the official Keras CIFAR dataset loaders.
- Headline results in this README come from the JSON files currently saved in `outputs/` and `cifar100/outputs/`.
- `src/train_improved.py` uses a stratified train/validation split and keeps the test set for final evaluation.
- Some earlier educational scripts are more demonstration-oriented than benchmark-oriented, so validation handling is not identical across every experiment.

## References

These are useful citations if the project is adapted into a report or dissertation chapter.

- Alex Krizhevsky. *Learning Multiple Layers of Features from Tiny Images*. 2009.
- Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. *Deep Residual Learning for Image Recognition*. 2016.
- Terrance DeVries, Graham W. Taylor. *Improved Regularization of Convolutional Neural Networks with Cutout*. 2017.
- Rafael Muller, Simon Kornblith, Geoffrey Hinton. *When Does Label Smoothing Help?*. 2019.
- Mingxing Tan, Quoc V. Le. *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*. 2019.
