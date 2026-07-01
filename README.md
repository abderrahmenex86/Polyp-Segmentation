# Endoscopy Polyp Segmentation

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white) ![MONAI](https://img.shields.io/badge/MONAI-%23000000.svg?style=for-the-badge&logo=MONAI&logoColor=white)

## Performance Diagnostics

*Still in the training phase*

## Core Features

- **Pure PyTorch Architecture**: Granular control via zero-wrapper training lifecycle.
- **MONAI Integration**: Hardened medical image augmentations, `DiceCELoss`, and standard DSC/IoU tracking.
- **Fault-Tolerant Inference**: Dynamically rebuilding network structures from `hyperparameters.json` artifacts.
- **Explicit Engineering**: Adheres strictly to the Ex86 blueprint zero-comment, non-abbreviated nomenclature.

## Domain Context

Combined robust feature extraction against highly imbalanced endoscopy datasets (Kvasir-SEG, CVC-ClinicDB, CVC-ColonDB, ETIS-LaribPolypDB, CVC-300). Implements aggressive spatial affine variations and Dice focal optimization to combat lighting/artifacting within standard gastrointestinal topology.

## Getting Started

```bash
git clone https://github.com/abderrahmenex86/Polyp-Segmentation
pip install torch torchvision torchaudio monai tqdm matplotlib
```

## CLI Execution Workflows

```bash
# 1. Verify combined datasets (assumes dataset/images and dataset/masks)
python tools.py --mode verify --path dataset

# 2. Train Model (UNet / UNETR / ViT / PraNet / MedSAM / Custom)
python main.py --train --architecture_name UNet --batch_size 16 --maximum_epochs 50 --learning_rate 0.0002

# 3. Test Best Model (Replace folder path)
python main.py --test --artifact_directory artifacts/20260701_004500_UNet

# 4. Infer on raw video frames
python main.py --infer --inference_directory dataset/unlabeled --artifact_directory artifacts/20260701_004500_UNet

# 5. Plot Artifact Results
python tools.py --mode plot --path artifacts/20260701_004500_UNet
```
