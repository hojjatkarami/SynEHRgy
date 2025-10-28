#!/bin/bash

source activate paper2025
cd /home/hokarami/code/SynEHRgy

# v2 old publised

# v3 new trainer
# python train.py run_name=synehrgy-mimic-v3 model_config=gpt2
# python generate.py run_name=synehrgy-mimic-v3

# source activate synehrgy_results
# python generate_results.py data_name=synehrgy-mimic-v3

# qwenn-small
# python train.py run_name=mimic3-qwen-s model_config=qwen-small
# python generate.py run_name=mimic3-qwen-s
# python generate_results.py data_name=mimic3-qwen-s





# pip install MIMIC_IV_MEDS
export DATASET_DOWNLOAD_USERNAME=ho-karami1
export DATASET_DOWNLOAD_PASSWORD=.2XkVCxa*_n*8sL
# MEDS_extract-MIMIC_IV root_output_dir="/home/hokarami/data/homes/hokarami/data/mimic4"


pip install eICU-MEDS # use `pip install -e .` for local installation in editing mode
pip install hydra-joblib-launcher --upgrade
export N_WORKERS=8


MEDS_extract-eICU root_output_dir="/home/hokarami/data/homes/hokarami/data/eicu_meds" do_download=True