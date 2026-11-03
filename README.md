# Endoscopy Polyp Segmentation

A comparison of convolutional, transformer, and parallel reverse-attention
segmentation models on endoscopy frames, in plain PyTorch. The point of the
repository is the benchmark: which architecture family actually holds up on a
small, specialised medical dataset.

![Comparative metrics](docs/figs/comparative_metrics_evaluation.png)

## Results

Validation scores over the best epoch of each run:

| Model | Parameters | Val DSC | Val IoU | Behaviour |
|---|---:|---:|---:|---|
| PraNet (Res2Net) | 24.48 M | 0.7628 | 0.6711 | Stable, best at epoch 98 |
| UNet (from scratch) | 1.83 M | 0.3722 | 0.2541 | Plateau at epoch 60 |
| UNetR (from scratch) | 105.32 M | 0.2941 | 0.1895 | Early stopped at epoch 70 |

PraNet beats both from-scratch models by a wide margin, and the transformer is
the worst of the three despite being about 57 times larger than the UNet. On a dataset this size, the inductive bias of a
convolutional decoder matters more than parameter count.

## What it does

- Trains UNet, UNetR, and PraNet behind one CLI, with the same data pipeline and
  metric recording.
- Profiles the hardware first, so batch size and image resolution fit the GPU
  before a long run starts.
- Saves per-run configuration and metric history under `artifacts/`, so any
  checkpoint can be re-tested without retraining.
- Exports inference overlays for a folder of frames.

## Run it

```bash
git clone https://github.com/abderrahmenex86/Polyp-Segmentation
cd Polyp-Segmentation
pip install -r requirements.txt

python main.py --profile --architecture PraNet --image_height 352 --image_width 352
python main.py --train --architecture PraNet
python main.py --test --run_dir artifacts/20260702_051722_PraNet
python main.py --infer --inference_dir dataset/unlabeled --run_dir artifacts/20260702_051722_PraNet
python tools.py --mode plot
```

Split the dataset into `train`, `val`, and `unlabeled` folders before training.

## Context

Endoscopy frames have specular highlights, motion blur, and low contrast at the
polyp boundary. Segmentation models trained on natural images transfer poorly,
which is why the benchmark includes a from-scratch UNet as a floor.

## Limits

- One dataset. Results on other endoscopy hardware are unknown.
- The two from-scratch models start without pretrained weights, so part of the
  gap may come from initialisation rather than architecture.
- Not a clinical tool.

## Stack

Python · PyTorch · MONAI · TorchMetrics · Optuna
