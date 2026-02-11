### Machine Learning HW 1
### Conents Within This Folder Include:
##Comparative analysis of MLP, CNN, and Vision Transformer architectures across tabular, natural image, and medical image datasets.

## Quick Start
```bash
# Install dependencies
pip install torch torchvision numpy pandas scikit-learn matplotlib seaborn tqdm

# Run experiments
python adult_main.py --model mlp --epochs 11        # Tabular data
python cifar10_main.py --model cnn --epochs 11     # Natural images
python pcam_main.py --model cnn --epochs 11         # Medical images
```

## Datasets
| **UCI Adult Income** | Binary classification (income >50K) | 48,842 samples | 14 features (6 num, 8 cat) | 2 | Accuracy, F1 |
| **CIFAR-100** | Multi-class object recognition | 60,000 images | 32×32 RGB | 10 | Accuracy |
| **PatchCamelyon** | Tumor detection | 327,000 patches | 96×96 RGB histopathology | 2 | Accuracy, F1, AUC |

## Architectures

### MLP
```
**Best for:** Tabular data | **Params:** 42K-2.8M depending on input size

### CNN
```
**Best for:** Image data | **Params:** 485K-612K

### Vision Transformer (Bonus)

# Deep Learning Architecture Comparison

Comparative analysis of MLP, CNN, and Vision Transformer architectures across tabular, natural image, and medical image datasets.

## Project Structure
```
project/
│
├── configs/
│   ├── adult_config.py          # Hyperparameters for UCI Adult dataset
│   ├── cifar10_config.py        # Hyperparameters for CIFAR-10 dataset
│   └── pcam_config.py           # Hyperparameters for PatchCamelyon dataset
│
├── datasets/
│   ├── adult_dataset.py         # Adult Income data loading & preprocessing
│   ├── cifar10_dataset.py       # CIFAR-10 data loading & augmentation
│   └── pcam_dataset.py          # PatchCamelyon data loading
│
├── models/
│   ├── mlp.py                   # Multilayer Perceptron implementation
│   ├── cnn.py                   # Convolutional Neural Network implementation
│   └── vit.py                   # Vision Transformer implementation (bonus)
│
├── train/
│   └── trainer.py               # Training loop with early stopping
│
├── eval/
│   └── evaluator.py             # Evaluation metrics & testing
│
├── utils/
│   ├── plot_curves.py           # Visualization utilities
│   └── timer.py                 # Training time tracking
│
├── adult_main.py                # Main script for Adult Income experiments
├── cifar10_main.py              # Main script for CIFAR-10 experiments
├── pcam_main.py                 # Main script for PatchCamelyon experiments
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended, optional for CPU training)

### Setup
```bash
# Clone repository
git clone <your-repo-url>
cd project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies (requirements)
```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
```

## Dataset Setup

### Dataset A: UCI Adult Income

**Download & Prepare:**
```bash
# Automatic download (handled by script)
python adult_main.py --download

# Or manual download from UCI repository:
# https://archive.ics.uci.edu/ml/datasets/adult
# Place adult.data and adult.test in datasets/data/adult/
```
**Data Location:**
```
datasets/data/adult/
├── adult.data     # Training data
```
**Preprocessing (handled automatically):**
- One-hot encoding for 8 categorical features
- StandardScaler normalization for 6 numerical features
- Train/Val/Test split: 70%/15%/15%

### Dataset B: CIFAR-10

**Download & Prepare:**
```bash
# Automatic download (handled by torchvision)
python cifar10_main.py --download

# Data will be saved to:
datasets/data/cifar10/
```
**Data Location:**
```
HW-1/Datasets/
```
**Preprocessing (handled automatically):**
- Normalization: mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
- Training augmentation: RandomCrop(32, padding=4), RandomHorizontalFlip
- Train/Val/Test split: 45K/5K/10K

### Dataset C: PatchCamelyon (PCam)

**Download & Prepare:**
```bash
# Download from official source
# https://github.com/basveeling/pcam

# Place files in:
datasets/data/pcam/
├── camelyonpatch_level_2_split_train_x.h5
├── camelyonpatch_level_2_split_train_y.h5
├── camelyonpatch_level_2_split_valid_x.h5
├── camelyonpatch_level_2_split_valid_y.h5
├── camelyonpatch_level_2_split_test_x.h5
└── camelyonpatch_level_2_split_test_y.h5
```
**Preprocessing (handled automatically):**
- Normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- No augmentation (preserve medical features)
- Uses predefined train/val/test splits

## Running Experiments

### Basic Usage
```bash
# UCI Adult Income - MLP
python adult_main.py --model mlp --epochs 11

# UCI Adult Income - CNN
python adult_main.py --model cnn --epochs 11

# CIFAR-10 - CNN
python cifar10_main.py --model cnn --epochs 11

# CIFAR-10 - Vision Transformer (bonus)
python cifar10_main.py --model vit --epochs 11

# PatchCamelyon - CNN
python pcam_main.py --model cnn --epochs 11
```

### Advanced Options
```bash
# Custom hyperparameters
python adult_main.py --model mlp --lr 0.0005 --batch_size 256 --epochs 50

# Enable GPU
python adult_main.py --model mlp --device cuda

```

## Configuration Files

Each dataset has a configuration file in `configs/` defining hyperparameters:

**Example: adult_config.py**
```python
config = {
    'learning_rate': 0.001,
    'batch_size': 128,
    'epochs': 50,
    'weight_decay': 1e-4,
    'dropout': 0.3,
    'early_stopping_patience': 7,
    'optimizer': 'adam',
    'scheduler': 'reduce_on_plateau'
}
```

Modify these files to experiment with different hyperparameters.

## Output Structure

After running experiments, results are saved to:
```
project/
│
├── configs/
│   ├── adult_config.py
|   |__ CIFAR-100.py
│   └── pcam_config.py
│
├── datasets/
│   ├── adult_dataset.py
|   |__ CIFAR-100_dataset.py
│   └── pcam_dataset.py
│
├── models/
│   ├── adult_model.py
|   |__ CIFAR-100_model.py
│   └── pcam_model.py
│
├── train/
│   └── trainer.py
│
├── eval/
│   └── evaluator.py
│
├── utils/
│   ├── plot_curves.py
│   └── timer.py
│
├── adult_main.py
|___ CIFAR-100_main.py
└── pcam_main.py

```




