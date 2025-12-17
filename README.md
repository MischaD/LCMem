# LCMem: A Universal Model for Robust Image Memorization Detection

This repository contains the official implementation of LCMem.

## Overview
LCMem is a two-stage training process for detecting memorization in generative models. 
1. **Stage 1**: Train a Siamese network using precomputed latents.
2. **Stage 2**: Fine-tune the network in image space with strong augmentations.

## Repository Structure
- `src/`: Helper scripts for latent computation and experiments.
- `src/privacy/`: Main source code including `main.py` for training and `networks/`.
- `00_compute_latents.sh`: Script to pre-compute latents for Stage 1.
- `03_convnext.sh`: Script to run the training/fine-tuning (Stage 2 example).
- `inference_example.py`: Example script to run inference using a trained model.
- `benchmark.py`: Script to benchmark the model against baselines.
- `benchmark_robustness.py`: Script to evaluate robustness.

## Setup
Ensure you have PyTorch, Transformers, Diffusers, and generic scientific computing libraries installed.

## Workflow

### 0. Data Preparation
Compute latents for your dataset.
```bash
bash 00_compute_latents.sh
```
*Note: You may need to adjust the `BASEDIR` and `FILELIST` paths in the script.*

### 1. Stage 1 Training
Train the base model using the computed latents.
Configuration: `src/privacy/config_OURS_final.json`

To run this, use `src/privacy/main.py` with the config.
Example command:
```bash
cd src/privacy
python main.py --config config_OURS_final.json --experiment_description Ours_Stage_one
```

### 2. Stage 2 Fine-tuning
Fine-tune the model in image space with strong data augmentation.
Configuration: `src/privacy/config_finetune_stronger_aug4.json`

Use the provided script or run manually:
```bash
# Using the script
bash 03_convnext.sh

# Or manually
cd src/privacy
python main.py --config config_finetune_stronger_aug4.json --experiment_description FineTune-ImageSpace-stronger4
```
*Note: Ensure `config_finetune_stronger_aug4.json` points to the correct Stage 1 checkpoint in the `model_path` field.*

## Inference
We provide an example script `inference_example.py` to demonstrate how to load the best performing model and run inference on an image.

```bash
python inference_example.py
```
This script demonstrates how to instantiate the `ImageSpaceSiameseDetector`.

## Benchmarking
You can run the full benchmark using:
```bash
python benchmark.py --filelist <path_to_lists> --basedir <path_to_images>
```


## Model availability

Since Stable Diffusion v2 is no longer publically available, we cannot release the model weights right now. We are working on a solution. 



## Citation
If you find this work useful, please cite our paper:
```bibtex
@misc{dombrowski2025lcmemuniversalmodelrobust,
      title={LCMem: A Universal Model for Robust Image Memorization Detection}, 
      author={Mischa Dombrowski and Felix Nützel and Bernhard Kainz},
      year={2025},
      eprint={2512.14421},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.14421}, 
}
```