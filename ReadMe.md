# [**SynEHRgy**: Synthesizing Mixed-Type Structured Electronic Health Records using Decoder-Only Transformers](https://arxiv.org/abs/2411.13428)

[![](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/Y-debug-sys/Diffusion-TS/blob/main/LICENSE)
<img src="https://img.shields.io/badge/python-3.9.7-blue">
<img src="https://img.shields.io/badge/pytorch-2.2.2-orange">

> **Abstract:** Generating synthetic Electronic Health Records (EHRs) offers significant potential for data augmentation, privacy-preserving data sharing, and improving machine learning model training. We propose a novel tokenization strategy tailored for structured EHR data, which encompasses diverse data types such as covariates, ICD codes, and irregularly sampled time series. Using a GPT-like decoder-only transformer model, we demonstrate the generation of high-quality synthetic EHRs. Our approach is evaluated using the MIMIC-III dataset, and we benchmark the fidelity, utility, and privacy of the generated data against state-of-the-art models.

## Contents

- [Installation](#installation)
- [Datasets](#datasets)
- [Quick Start](#quick-start)
<!-- - [Citation](#citation) -->

## Installation

Clone the repository, create a virtual environment (`venv` or `conda`), and install the required packages using `pip`:

```bash
# clone the repository
git clone https://github.com/hojjatkarami/SynEHRgy.git
cd SynEHRgy

# using virtualenv
python3 -m venv synehrgy
source synehrgy/bin/activate

# OR using conda
conda create --name synehrgy python=3.9.7 --yes
conda activate synehrgy

# install the required packages
pip install -r requirements.txt
```

## Datasets

We use MIMIC-III dataset containing structured EHR data of approximately 42,000 patients. After preprocessing, we have 4,656 unique ICD codes, 41 irregularly-sampled time series from vital signs and laboratory variables, and a set of covariates. Please refer to the [data](data) folder for more details on the datasets.

## Quick Start

We use `hydra-core` library for managing all configuration parameters. You can change them from [config](configs) folder.

We highly recommend using `wandb` for logging and tracking the experiments. Get your API key from [wandb](https://wandb.ai/authorize). Create a `.env` file in the root directory and add the following line:

```bash
WANDB_API_KEY=your_api_key
```

### Training

The SynEHRgy model can easily be trained using the following command:

```bash
python train.py hparams.n_ctx=1024 hparams.mini_batch=64 run_name='synehrgy-mimic' data=mimic3 preprocess.bin_type=uniform model=gpt soft_labels=False
```

The configuration file is located at [`configs/configTrain.yaml`](configs/configTrain.yaml). The model will be saved at `saved_models/{MODEL_NAME}`.

### Generation

To generate synthetic data, you can use the following command:

```bash
python generate.py 'model="synehrgy-mimic"' n_samples=30000 bin_type=uniform fix_covars=False batch_size=1024
```

This will generate 30,000 synthetic patients using the trained model `synehrgy-mimic-hard-uni[v1]` and save the results in the ['data/synthetic'](data/synthetic/) folder. The configuration file is located at [`configs/configGenerate.yaml`](configs/configGenerate.yaml).

**Alternatively, you can use the jupyter notebook ['Tutoria.ipynb'](Tutoria.ipynb) for a follow-along tutorial.**

### Evaluation

To replicate the results in the paper, you can use ['Results.ipynb'](Results.ipynb) notebook. The results will be saved in ['Results'](Results/) folder.

## Citation

If you find this repo useful, please cite our paper via

```bibtex
@inproceedings{karamisynehrgy,
  title={SynEHRgy: Synthesizing Mixed-Type Structured Electronic Health Records using Decoder-Only Transformers},
  author={Karami, Hojjat and Atienza, David and Paraschiv-Ionescu, Anisoara},
  booktitle={GenAI for Health: Potential, Trust and Policy Compliance}
}
```
