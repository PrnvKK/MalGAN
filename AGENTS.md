# MalGAN — AI Agent Guide

## Project Overview

GAN-based data augmentation pipeline for malware image classification (MaleVis dataset). Trains a baseline ResNet50 classifier on 6 malware families, trains a Conditional DCGAN to generate synthetic malware images, then retrains the classifier on real + GAN-augmented data and compares performance.

## Repository Structure

| File | Purpose |
|---|---|
| `load_data.py` | Reads MaleVis dataset (pre-split train/val folders) via OpenCV, returns numpy arrays |
| `load_balanced_data.py` | Selects 6 families (Androm, Elex, Expiro, HackKMS, Hlux, Sality) into a balanced subset |
| `cnn_arch.py` | Defines ResNet50 (transfer learning) and custom CNN architectures; compiles model |
| `cnn_baseline_train.py` | Trains baseline ResNet50 on balanced 6-class data; saves history/plots/predictions |
| `gan_arch.py` | Defines Conditional DCGAN Generator (64x64) and Discriminator; builds combined GAN |
| `gan_train.py` | Full GAN training loop with adaptive D/G ratio, label smoothing, LR decay, NaN detection |
| `gan_augment.py` | Post-training: loads generator, produces synthetic images per class, upscales 64→224 |
| `cnn_train_augmented.py` | Loads original + GAN-augmented data, trains identical ResNet50, saves results |

## Key Details

- **Target classes:** Androm, Elex, Expiro, HackKMS, Hlux, Sality (6 classes)
- **Input size:** 224×224 for CNN classifiers; 64×64 for GAN (upscaled back to 224)
- **Baseline CNN:** ResNet50 (ImageNet weights, last 20 layers trainable) + GAP → Dense(256) → Dense(6, softmax)
- **Generator:** Latent dim 100, label embedding(6→50), Conv2DTranspose stack 8→16→32→64
- **Discriminator:** Label embedding reshaped to 64×64 spatial map, concatenated with image, Conv2D downstack
- **GAN training:** 100 epochs, batch 64, adaptive D/G ratio (running avg window 10, target D acc 65–80%), label smoothing (0.9/0.1), gradient clipping (clipnorm=1.0), LR decay (linear after epoch 50)
- **Augmentation:** 100 synthetic images generated per class
- **Augmented CNN:** Same architecture/hyperparams as baseline, traind on real + synthetic combined

## Colab Data Paths

- MaleVis dataset: `/content/malevis_data/malevis_train_val_300x300`
- Project output: `/content/drive/MyDrive/GAN_Malware_Detection`
- Augmented images: `/content/augmented_data`

## Dependencies (inferred)

`tensorflow` (keras), `numpy`, `matplotlib`, `opencv-python`, `pathlib`, `tqdm`, `IPython`

## Output Files

| File | Source |
|---|---|
| `subset_samples.png` | `load_balanced_data.py` |
| `baseline_cnn_best.h5` | `cnn_baseline_train.py` |
| `baseline_training_history.json` | `cnn_baseline_train.py` |
| `baseline_training_curves.png` | `cnn_baseline_train.py` |
| `baseline_val_predictions.npy` | `cnn_baseline_train.py` |
| `baseline_val_true_labels.npy` | `cnn_baseline_train.py` |
| `gan_training/checkpoints/generator_final.h5` | `gan_train.py` |
| `gan_training/checkpoints/discriminator_final.h5` | `gan_train.py` |
| `gan_training/checkpoints/gan_final.h5` | `gan_train.py` |
| `gan_training/training_history.json` | `gan_train.py` |
| `gan_training/training_curves.png` | `gan_train.py` |
| `gan_training/final_samples_grid.png` | `gan_train.py` |
| `augmented_cnn_best.h5` | `cnn_train_augmented.py` |
| `augmented_training_history.json` | `cnn_train_augmented.py` |
| `augmented_val_predictions.npy` | `cnn_train_augmented.py` |
