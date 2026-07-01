# Endoscopy Polyp Segmentation

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white) ![MONAI](https://img.shields.io/badge/MONAI-%23000000.svg?style=for-the-badge&logo=MONAI&logoColor=white)

## Performance Diagnostics

*Still in the training phase*

Prior to every training run, the system automatically generates a diagnostic overlay plot (`docs/figs/pre_train_*.png`) from the augmented `CacheDataset`. This ensures that orientation and spacing transforms have not deformed the polyp morphology before committing to compute.

## Core Features

- **Hardware-Saturated DataLoaders**: Utilizes MONAI `CacheDataset` loaded into RAM, with 8 train workers, 4 validation/test workers, and a `prefetch_factor=4`.
- **Dynamic Hardware Profiling**: Use the `--profile` flag to automatically run dummy tensors through forward/backward passes to discover your exact GPU's Maximum Safe Batch Size, preventing mid-run OOM crashes.
- **BFloat16 AMP**: Eliminates the need for standard float16 `GradScaler` operations by using `bfloat16` for native dynamic range preservation during backpropagation.
- **Native Network Integrations**: Features a pure PyTorch implementation of PraNet (using the precise Res2Net bottle2neck backbone) and MedSAM, with explicit CLI hooks for loading specific backbone and architecture weights.
- **Explicit Serialization**: Every hyperparameter is strictly parsed via `argparse` and serialized into a precise `hyperparameters.json` alongside `architecture.txt`, allowing seamless state resumption and smart inference.

## Domain Context

Targeting highly imbalanced polyp segmentation in mixed endoscopy datasets (**Kvasir-SEG**, **CVC-ClinicDB**, **CVC-ColonDB**, **ETIS-LaribPolypDB**, **CVC-300**). Variable physical resolutions are normalized via uniform `Spacingd` and `Orientationd` transforms. Optimization is strictly managed using `DiceCELoss` paired with a linear warmup to cosine annealing learning rate scheduler.

## Getting Started

```bash
# Clone the repository
git clone https://github.com/abderrahmenex86/Polyp-Segmentation
cd Polyp-Segmentation

# Install dependencies
pip install torch torchvision torchaudio monai tqdm matplotlib segment-anything

# Prepare Dataset Directory Structure
# dataset/images/
# dataset/masks/
```

## CLI Execution Workflows

### 1. Hardware Profiling

*Automatically determine the maximum batch size your GPU can handle before initiating training.*

```bash
python main.py --profile --architecture PraNet --image_height 352 --image_width 352
```

### 2. Full Training Pipeline (Example: PraNet)

*Execute with pre-trained Res2Net and PraNet weights.*

```bash
python main.py --train \
--architecture PraNet \
--batch_size 16 \
--epochs 100 \
--warmup_epochs 10 \
--learning_rate 0.0001 \
--dataset_directory dataset \
--backbone_weights weights/Res2Net50_26w_4s.pth \
--pranet_weights weights/PraNet.pth
```

### 3. Test Best Model

*Evaluate the model using the serialized best weights and hyperparameter config.*

```bash
python main.py --test --run_dir artifacts/20260701_004500_PraNet
```

### 4. Smart Inference

*Run raw frames through the dynamically rebuilt network.*

```bash
python main.py --infer --inference_dir dataset/unlabeled --run_dir artifacts/20260701_004500_PraNet
```

### 5. Plot Generation

*Generate visual learning curves from the serialized epoch history.*

```bash
python tools.py --mode plot --path artifacts/20260701_004500_PraNet
```
