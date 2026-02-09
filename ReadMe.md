# ⚠️ Deprecated Repository

**This repository is deprecated and no longer maintained.**

➡️ **Please use the new version instead:**  
👉 https://github.com/hojjatkarami/SynEHRgy-v2

All future development, bug fixes, and updates happen in the new repository.

---
# SynEHRgy: Synthesizing Mixed-Type Structured Electronic Health Records

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/Y-debug-sys/Diffusion-TS/blob/main/LICENSE)
[![Python 3.9.7](https://img.shields.io/badge/python-3.9.7-blue)](https://www.python.org/downloads/)
[![PyTorch 2.2.2](https://img.shields.io/badge/pytorch-2.2.2-orange)](https://pytorch.org/)
[![Paper](https://img.shields.io/badge/arXiv-2411.13428-b31b1b.svg)](https://arxiv.org/abs/2411.13428)

A decoder-only transformer model for generating high-quality synthetic Electronic Health Records (EHRs) using a novel tokenization strategy tailored for mixed-type structured data.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Dataset](#dataset)
- [Quick Start](#quick-start)
  - [Training](#training)
  - [Generation](#generation)
  - [Evaluation](#evaluation)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [License](#license)

## Overview

**SynEHRgy** generates synthetic Electronic Health Records using GPT-like decoder-only transformers. This approach addresses critical needs in healthcare AI:

- **Data Augmentation**: Expand limited medical datasets for better model training
- **Privacy Preservation**: Share realistic data without exposing patient information
- **Research Enablement**: Provide accessible datasets for healthcare ML research

Our model handles diverse EHR data types including covariates, ICD codes, and irregularly sampled time series from vital signs and laboratory measurements.

## Key Features

✨ **Novel Tokenization Strategy** for mixed-type structured EHR data  
🏥 **Trained on MIMIC-III** dataset with ~42,000 patients  
📊 **Handles Multiple Data Types**: covariates, ICD codes, irregular time series  
🎯 **High-Quality Generation**: Benchmarked for fidelity, utility, and privacy  
⚙️ **Flexible Configuration** using Hydra framework  

## Installation

### Prerequisites

- Python 3.9.7+
- Conda (recommended) or venv
- CUDA-compatible GPU (recommended for training)

### Setup

```bash
# Clone the repository
git clone https://github.com/hojjatkarami/SynEHRgy.git
cd SynEHRgy

# Create and activate conda environment (recommended)
conda env create -f synehrgy.yaml
conda activate synehrgy

# install synehrgy
pip install -e .


```

### Configure Weights & Biases (Optional but Recommended)

For experiment tracking and logging:

1. Get your API key from [wandb.ai/authorize](https://wandb.ai/authorize)
2. Create a `.env` file in the root directory:

```bash
echo "WANDB_API_KEY=your_api_key_here" > .env
```

## Dataset

We use the **MIMIC-III** dataset containing structured EHR data from approximately 42,000 patients.

**Preprocessed Data Includes:**
- 4,656 unique ICD diagnostic codes
- 41 irregularly-sampled time series (vital signs and lab variables)
- Patient demographic and clinical covariates

📂 See the [`data/`](data) folder for detailed dataset information and preprocessing steps.

## Quick Start

All configuration is managed through [Hydra](https://hydra.cc/). Default settings are in the [`configs/`](configs) folder.

### Training

Train the SynEHRgy model with default parameters:

```bash
python train.py \
    hparams.n_ctx=256 \
    hparams.mini_batch=64 \
    run_name='synehrgy-mimic' \
    data=mimic3 \
    preprocess.bin_type=uniform \
    model=gpt
```

**Key Parameters:**
- `n_ctx`: Context window size (default: 256)
- `mini_batch`: Batch size for training
- `run_name`: Experiment name for tracking
- `bin_type`: Time series binning strategy (`uniform` or `quantile`)

**Output:** Model checkpoints saved to `saved_models/{MODEL_NAME}/`

**Configuration:** [`configs/configTrain.yaml`](configs/configTrain.yaml)

### Generation

Generate synthetic patient records:

```bash
python generate.py \
    run_name="synehrgy-mimic" \
    n_samples=30000 \
    bin_type=uniform \
    fix_covars=False \
    batch_size=1024
```

**Parameters:**
- `n_samples`: Number of synthetic patients to generate (default: 30,000)
- `run_name`: Name of the run
- `fix_covars`: Whether to fix covariates during generation
- `batch_size`: Generation batch size

**Output:** Synthetic data saved to [`data/synthetic/`](data/synthetic/)

**Configuration:** [`configs/configGenerate.yaml`](configs/configGenerate.yaml)

### Evaluation

First, create a new environment as `synthcity` module have compatibility issues with the latest versions of some libraries.

```bash
conda create -n synehrgy_results python=3.12
conda activate synehrgy_results
pip install synthcity==0.2.12 Levenshtein
pip install ipykernel omegaconf plotly nbformat>=4.2.0
pip install --upgrade kaleido
pip install opacus==1.5.3
pip install openTSNE



conda env create -f synehrgy_results.yaml
conda activate synehrgy_results

# install synehrgy
pip install -e .
```

Evaluate generated data quality using the provided notebook:

```bash
jupyter notebook Results.ipynb
```

This notebook reproduces all results from the paper, including:
- **Fidelity Metrics**: Statistical similarity to real data
- **Utility Metrics**: Downstream task performance
- **Privacy Metrics**: Patient re-identification risk

**Output:** Results and visualizations saved to [`Results/`](Results/)

## Project Structure

```
SynEHRgy/
├── configs/              # Hydra configuration files
│   ├── configTrain.yaml
│   └── configGenerate.yaml
├── data/                 # Dataset folder
│   └── synthetic/        # Generated synthetic data
├── saved_models/         # Trained model checkpoints
├── Results/              # Evaluation results and plots
├── train.py              # Training script
├── generate.py           # Generation script
├── Tutorial.ipynb        # Interactive tutorial
├── Results.ipynb         # Evaluation notebook
├── env.yaml              # Conda environment file
└── ReadMe.md             # This file
```

## Citation

If you use SynEHRgy in your research, please cite our paper:

```bibtex
@inproceedings{karamisynehrgy,
  title={SynEHRgy: Synthesizing Mixed-Type Structured Electronic Health Records using Decoder-Only Transformers},
  author={Karami, Hojjat and Atienza, David and Paraschiv-Ionescu, Anisoara},
  booktitle={GenAI for Health: Potential, Trust and Policy Compliance},
  year={2024}
}
```

**Paper:** [https://arxiv.org/abs/2411.13428](https://arxiv.org/abs/2411.13428)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Questions or Issues?** Please open an issue on [GitHub](https://github.com/hojjatkarami/SynEHRgy/issues).
