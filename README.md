# Human Activity Recognition Dissertation

## Project Overview

This repository contains the implementation developed as part of the **MSc Software Engineering dissertation** at the **University of West London**.

The project investigates the effectiveness of **transfer learning-based Convolutional Neural Network (CNN) architectures** for **image-based Human Activity Recognition (HAR)**.

The study compares two pretrained CNN architectures:

- **ResNet50**
- **MobileNetV3-Large**

The models are evaluated using different training strategies and data augmentation conditions to investigate classification performance and computational efficiency.

---

## Dissertation Title

**A Comparative Study of Transfer Learning-Based CNN Architectures for Image-Based Human Activity Recognition**

---

## Author

**Ujjwal Shrestha**  
MSc Software Engineering  
University of West London

---

## Research Focus

The experimental study focuses on:

- Transfer learning for image-based Human Activity Recognition
- Comparison of ResNet50 and MobileNetV3-Large
- Feature extraction versus fine-tuning
- The impact of data augmentation
- Classification performance
- Computational efficiency
- Model size and parameter count
- Inference time

The project uses the **Stanford40 Actions Dataset**, which contains 40 human activity categories.

---

# Experimental Design

Eight controlled experiments were conducted.

Each experiment combines:

1. CNN architecture
2. Training strategy
3. Data augmentation condition

| Experiment | Architecture | Training Strategy | Data Augmentation |
|------------|--------------|-------------------|-------------------|
| 1 | ResNet50 | Feature Extraction | No |
| 2 | ResNet50 | Feature Extraction | Yes |
| 3 | ResNet50 | Fine-Tuning | No |
| 4 | ResNet50 | Fine-Tuning | Yes |
| 5 | MobileNetV3-Large | Feature Extraction | No |
| 6 | MobileNetV3-Large | Feature Extraction | Yes |
| 7 | MobileNetV3-Large | Fine-Tuning | No |
| 8 | MobileNetV3-Large | Fine-Tuning | Yes |

All experiments use the same dataset, official Stanford40 train/test split, validation split strategy, image size, batch size, number of epochs, and evaluation metrics.

---

# Technologies and Requirements

## Software Requirements

The project uses:

- Python 3.11.x
- PyTorch
- Torchvision
- Scikit-learn
- Matplotlib
- Pillow
- NumPy
- Git

**Visual Studio Code** is recommended for development.

---

# Project Structure

The project follows a modular structure:

```text
HAR_Dissertation/
│
├── dataset/
│   ├── images/
│   ├── ImageSplits/
│   └── preprocessed/
│       └── images/
│
├── saved_models/
│   ├── resnet_feature_extraction.pth
│   ├── resnet_feature_extraction_augmentation.pth
│   ├── resnet_fine_tuning.pth
│   ├── resnet_fine_tuning_augmentation.pth
│   ├── mobilenet_feature_extraction.pth
│   ├── mobilenet_feature_extraction_augmentation.pth
│   ├── mobilenet_fine_tuning.pth
│   └── mobilenet_fine_tuning_augmentation.pth
│
├── results/
│   ├── confusion_matrices/
│   ├── computational_efficiency.csv
│   ├── final_experiment_comparison.csv
│   ├── test_set_evaluation.csv
│   └── experiment_logs.csv
│
├── src/
│   ├── config/
│   │   ├── constants.py
│   │   └── paths.py
│   │
│   ├── models/
│   │   ├── resnet.py
│   │   ├── mobilenet.py
│   │   └── ...
│   │
│   └── training/
│       ├── dataset.py
│       ├── dataloader.py
│       ├── transforms.py
│       ├── train.py
│       ├── evaluate.py
│       ├── evaluate_models.py
│       ├── analyse_efficiency.py
│       ├── compare_experiments.py
│       └── ...
│
├── run_experiment.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

# Installation

## 1. Clone the Repository

### SSH

```bash
git clone git@github.com:ujjwal-27/HAR_dissertation.git
cd HAR_dissertation
```

### HTTPS

```bash
git clone https://github.com/ujjwal-27/HAR_dissertation.git
cd HAR_dissertation
```

---

## 2. Create a Virtual Environment

Python 3.11 is recommended.

```bash
python3.11 -m venv .venv
```

---

## 3. Activate the Virtual Environment

### macOS / Linux

```bash
source .venv/bin/activate
```

### Windows Command Prompt

```cmd
.venv\Scripts\activate
```

### Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

---

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 5. Verify Python

```bash
python --version
```

Expected output:

```text
Python 3.11.x
```

---

# Dataset

## Stanford40 Actions Dataset

This project uses the **Stanford40 Actions Dataset** for image-based Human Activity Recognition.

The dataset contains:

- **40 activity classes**
- **9,532 images**

The dataset is **not included in this repository**.

Users must obtain the dataset separately and organise it according to the structure required by the project.

---

## Dataset Organisation

The project uses the official Stanford40 split files located in:

```text
dataset/ImageSplits/
```

The images used by the training pipeline are organised by activity class under:

```text
dataset/preprocessed/images/
```

The expected structure is:

```text
dataset/
│
├── images/
│
├── ImageSplits/
│   ├── applauding_train.txt
│   ├── applauding_test.txt
│   ├── blowing_bubbles_train.txt
│   ├── blowing_bubbles_test.txt
│   └── ...
│
└── preprocessed/
    └── images/
        ├── applauding/
        ├── blowing_bubbles/
        ├── brushing_teeth/
        ├── ...
        └── 40 activity classes
```

The official split files determine which images belong to the training and testing sets.

---

# Dataset Analysis

Dataset analysis can be performed using:

```bash
python src/data/dataset_analysis.py
```

The analysis checks dataset characteristics including:

- Number of valid images
- Corrupted images
- Image dimensions
- Image formats
- Colour modes
- Class distribution

---

# Image Preprocessing

The CNN models operate on images with a target size of:

```text
224 × 224 pixels
```

Images are:

1. Loaded using Pillow
2. Converted to RGB
3. Converted to PyTorch tensors
4. Normalised using ImageNet mean and standard deviation

The ImageNet normalisation values used are:

```text
Mean:
[0.485, 0.456, 0.406]

Standard Deviation:
[0.229, 0.224, 0.225]
```

---

# Data Augmentation

Data augmentation is applied **only to the training data** when enabled.

The implemented augmentation pipeline consists of:

- Random horizontal flipping
- Random rotation up to ±15 degrees
- Random colour jittering

Colour jittering modifies:

- Brightness
- Contrast
- Saturation

The training pipeline with augmentation is:

```text
Image
  │
  ├── Random Horizontal Flip
  │
  ├── Random Rotation (±15°)
  │
  ├── Colour Jitter
  │     ├── Brightness
  │     ├── Contrast
  │     └── Saturation
  │
  ├── ToTensor
  │
  └── ImageNet Normalisation
```

When augmentation is disabled, the training pipeline is:

```text
Image
  │
  ├── ToTensor
  │
  └── ImageNet Normalisation
```

Validation and test data do **not** use random augmentation. They use only:

```text
Image
  │
  ├── ToTensor
  │
  └── ImageNet Normalisation
```

This ensures that validation and test evaluation remains deterministic.

---

# Dataset Splitting

The official Stanford40 training split is divided into:

- **80% training**
- **20% validation**

The official Stanford40 test split is retained as the independent test set.

A fixed random seed is used to make the training/validation split reproducible:

```text
Random Seed = 42
```

---

# Model Architectures

## ResNet50

The project uses the pretrained **ResNet50** architecture provided by Torchvision.

The original final classification layer is replaced with a new classifier containing:

```text
40 output classes
```

Two training strategies are evaluated.

### Feature Extraction

The pretrained ResNet50 parameters are frozen.

Only the newly added classification layer is trained.

### Fine-Tuning

All ResNet50 parameters are made trainable and updated during training.

---

## MobileNetV3-Large

The project uses the pretrained **MobileNetV3-Large** architecture provided by Torchvision.

The original final classification layer is replaced with a new classifier containing:

```text
40 output classes
```

Two training strategies are evaluated.

### Feature Extraction

The pretrained feature-extraction layers are frozen.

Only the classification layer is trained.

### Fine-Tuning

All MobileNetV3-Large parameters are made trainable and updated during training.

---

# Training Configuration

The main training configuration is defined in:

```text
src/config/constants.py
```

| Parameter | Value |
|-----------|-------|
| Dataset | Stanford40 |
| Number of classes | 40 |
| Image size | 224 × 224 |
| Batch size | 32 |
| Validation split | 20% |
| Epochs | 20 |
| Learning rate | 0.001 |
| Optimizer | Adam |
| Loss function | Cross Entropy Loss |
| Random seed | 42 |

The project automatically selects the available computation device:

1. Apple Metal Performance Shaders (MPS)
2. CUDA
3. CPU

---

# Running an Experiment

Experiments are configured through:

```text
src/config/constants.py
```

## Model

For ResNet50:

```python
MODEL_NAME = "resnet"
```

For MobileNetV3-Large:

```python
MODEL_NAME = "mobilenet"
```

## Training Strategy

Feature extraction:

```python
TRAINING_MODE = "feature_extraction"
```

Fine-tuning:

```python
TRAINING_MODE = "fine_tuning"
```

## Data Augmentation

Without augmentation:

```python
USE_DATA_AUGMENTATION = False
```

With augmentation:

```python
USE_DATA_AUGMENTATION = True
```

The experiment name and output filenames are generated automatically from these configuration values.

---

# Run Training

From the project root:

```bash
python run_experiment.py
```

For example:

```python
MODEL_NAME = "resnet"
TRAINING_MODE = "feature_extraction"
USE_DATA_AUGMENTATION = True
```

runs:

```text
resnet_feature_extraction_augmentation
```

---

# Training Outputs

Each experiment produces a trained model checkpoint and a training log.

## Model Checkpoints

Model checkpoints are saved in:

```text
saved_models/
```

For example:

```text
saved_models/resnet_feature_extraction_augmentation.pth
```

The checkpoint contains the model parameters from the epoch that achieved the best validation accuracy.

## Training Logs

Training logs are saved in:

```text
results/
```

Each experiment has a corresponding CSV file containing:

- Epoch
- Training loss
- Training accuracy
- Validation loss
- Validation accuracy

For example:

```text
results/resnet_feature_extraction_augmentation.csv
```

---

# Test-Set Evaluation

After training all eight experiments, the saved models can be evaluated using the official Stanford40 test set.

Run:

```bash
python -m src.training.evaluate_models
```

The evaluation calculates:

- Accuracy
- Weighted Precision
- Weighted Recall
- Weighted F1-score

The results are saved to:

```text
results/test_set_evaluation.csv
```

The evaluation also generates confusion matrices for all eight experiments.

---

# Confusion Matrices

Confusion matrices are generated from the predictions produced during test-set evaluation.

They are saved in:

```text
results/confusion_matrices/
```

The directory contains one matrix for each experiment:

```text
resnet_feature_extraction.png
resnet_feature_extraction_augmentation.png
resnet_fine_tuning.png
resnet_fine_tuning_augmentation.png

mobilenet_feature_extraction.png
mobilenet_feature_extraction_augmentation.png
mobilenet_fine_tuning.png
mobilenet_fine_tuning_augmentation.png
```

---

# Computational Efficiency Analysis

Computational efficiency can be analysed using:

```bash
python -m src.training.analyse_efficiency
```

The analysis records:

- Total number of parameters
- Number of trainable parameters
- Model size
- Inference time per image

The results are saved to:

```text
results/computational_efficiency.csv
```

---

# Final Experiment Comparison

The test-set performance and computational-efficiency results can be combined into a single comparison table.

Run:

```bash
python -m src.training.compare_experiments
```

The resulting file is:

```text
results/final_experiment_comparison.csv
```

The final comparison contains:

- Experiment
- Architecture
- Training strategy
- Data augmentation
- Accuracy
- Precision
- Recall
- F1-score
- Total parameters
- Trainable parameters
- Model size
- Inference time per image

---

# Complete Experimental Workflow

To reproduce the complete experimental workflow:

### 1. Configure an experiment

Edit:

```text
src/config/constants.py
```

### 2. Train the model

```bash
python run_experiment.py
```

### 3. Repeat for all eight configurations

```text
1. ResNet50 + Feature Extraction + No Augmentation
2. ResNet50 + Feature Extraction + Augmentation
3. ResNet50 + Fine-Tuning + No Augmentation
4. ResNet50 + Fine-Tuning + Augmentation

5. MobileNetV3-Large + Feature Extraction + No Augmentation
6. MobileNetV3-Large + Feature Extraction + Augmentation
7. MobileNetV3-Large + Fine-Tuning + No Augmentation
8. MobileNetV3-Large + Fine-Tuning + Augmentation
```

### 4. Evaluate all trained models

```bash
python -m src.training.evaluate_models
```

### 5. Analyse computational efficiency

```bash
python -m src.training.analyse_efficiency
```

### 6. Generate the final comparison

```bash
python -m src.training.compare_experiments
```

---

# Final Experimental Outputs

The main generated results are stored in the `results/` directory:

```text
results/
│
├── confusion_matrices/
│   ├── resnet_feature_extraction.png
│   ├── resnet_feature_extraction_augmentation.png
│   ├── resnet_fine_tuning.png
│   ├── resnet_fine_tuning_augmentation.png
│   ├── mobilenet_feature_extraction.png
│   ├── mobilenet_feature_extraction_augmentation.png
│   ├── mobilenet_fine_tuning.png
│   └── mobilenet_fine_tuning_augmentation.png
│
├── computational_efficiency.csv
├── final_experiment_comparison.csv
├── test_set_evaluation.csv
│
└── experiment training logs
```

---

# Saved Model Checkpoints

The final experimental model checkpoints are stored in:

```text
saved_models/
```

The eight checkpoints are:

```text
resnet_feature_extraction.pth
resnet_feature_extraction_augmentation.pth
resnet_fine_tuning.pth
resnet_fine_tuning_augmentation.pth

mobilenet_feature_extraction.pth
mobilenet_feature_extraction_augmentation.pth
mobilenet_fine_tuning.pth
mobilenet_fine_tuning_augmentation.pth
```

---

# Reproducibility

The implementation uses a fixed random seed:

```text
42
```

The same dataset split, model configuration and software environment should therefore produce reproducible experimental conditions.

Results may still vary between hardware and software environments, particularly when hardware-specific acceleration such as Apple MPS or CUDA is used.

---

# Academic Purpose

This repository was developed as part of the **MSc Software Engineering dissertation** at the **University of West London**.

The implementation supports the experimental investigation of transfer learning-based CNN architectures for image-based Human Activity Recognition.

The repository contains the source code, experiment configurations, trained model checkpoints and generated experimental results used in the study.

---

# Dataset Availability

The Stanford40 dataset is not redistributed as part of this repository.

Users must obtain the dataset separately and comply with the applicable terms and conditions associated with its use.