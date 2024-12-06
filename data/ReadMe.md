# Data Processing

## Processing MIMIC-III Data

We use the preprocessing pipeline provided by [https://github.com/YerevaNN/mimic3-benchmarks](https://github.com/YerevaNN/mimic3-benchmarks). This pipeline is used to extract the structured EHR data of approximately 42,000 patients from the MIMIC-III dataset. We will use the exact same train-val-test split from the the following downstream tasks:

- In-hospital mortality prediction
- Phenotyping

Please note that in the original pipeline, only 17 time series variables are used. However, we will use 41 time series in our study. Please use the modifed variable mapping (`SynEHRgy_itemid_to_variable_map.csv`).

You can use this script:

```bash

cd data
git clone https://github.com/YerevaNN/mimic3-benchmarks.git

# Extracting the structured data
python -m mimic3benchmark.scripts.extract_episodes_from_subjects data/root4/ --variable_map_file SynEHRgy_itemid_to_variable_map.csv

# Splitting the data into train and test
python -m mimic3benchmark.scripts.split_train_and_test data/root4/

# Creating the phenotyping and in-hospital mortality datasets
python -m mimic3benchmark.scripts.create_phenotyping data/root4/ data/phenotyping2/
python -m mimic3benchmark.scripts.create_in_hospital_mortality data/root4/ data/in-hospital-mortality2/

# Splitting the data into train and validation
python -m mimic3models.split_train_val data/phenotyping2/
python -m mimic3models.split_train_val data/in-hospital-mortality2/

```

You need to specify the path to above directory in the jupyter notebook `prepare_mimic.ipynb` and run the cells to preprocess the data and save it in the [`processed`](processed) folder. Then you should create a model configuration file in the [`../configs/data`](../configs/data) folder.

## Synthetic Data

When you run the `generate.py` script, the synthetic data (time series+icd codes+covariates) will be saved in the [`synthetic`](synthetic) folder in a `.pkl` format.

## Synthetic time series data

When running the RESULTS.ipynb notebook, the synthetic time series data will be saved in the [`SynTS`](SynTS) folder

## Downloading the data

If you want to use the same data as we did, please send me an email ([g.hojatkarami@gmail.com](mailto:g.hojatkarami@gmail.com)) with your certificate of completion of the CITI training and I will provide you with the processed data.

Additinally, if you need the synthetic data of baseline models (TimEHR, RTSGAN, PromptEHR, HALO), do not hesitate to contact me.

## How to use your own data?

Currently, the code is designed to work with the MIMIC-III dataset. If you want to use your own data, you need to follow the same structure as the MIMIC-III dataset. You can inspect the processed datasets and change the code accordingly. If you need help, do not hesitate to contact me.
