"""
Example script showing how to load and use the processed data.

This script demonstrates:
1. Loading the metadata and discretized datasets
2. Inspecting the data structure
3. Basic statistics and visualizations
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def load_metadata(path_data: str, bin_type: str = 'uniform'):
    """Load metadata file."""
    metadata_file = f"{path_data}/metadata2.pkl" if bin_type == 'uniform' else f"{path_data}/metadata.pkl"
    print(f"Loading metadata from {metadata_file}...")
    
    with open(metadata_file, "rb") as f:
        metadata = pickle.load(f)
    
    print(f"✓ Loaded metadata")
    return metadata


def load_dataset(path_data: str, split: str = 'train', bin_type: str = 'uniform'):
    """Load discretized dataset."""
    dataset_file = f"{path_data}/{split}DiscDataset.pkl" if bin_type == 'uniform' \
                   else f"{path_data}/{split}DiscDatasetQuant.pkl"
    print(f"Loading {split} dataset from {dataset_file}...")
    
    with open(dataset_file, "rb") as f:
        data = pickle.load(f)
    
    print(f"✓ Loaded {len(data)} patients")
    return data


def inspect_metadata(metadata):
    """Print metadata statistics."""
    print("\n" + "=" * 80)
    print("METADATA STATISTICS")
    print("=" * 80)
    
    print(f"\nTotal tokens: {len(metadata['token2id'])}")
    print(f"Total variables: {len(metadata['var2id'])}")
    print(f"Total codes: {len(metadata['codeToId'])}")
    
    print("\nVocabulary sizes:")
    for k, v in metadata['vocab_size'].items():
        print(f"  {k}: {v}")
    
    print("\nVariable mappings:")
    for var_name, var_id in list(metadata['var2id'].items())[:5]:
        print(f"  {var_name}: {var_id}")
    print("  ...")
    
    print("\nSample codes:")
    for code, code_id in list(metadata['codeToId'].items())[:5]:
        print(f"  {code}: {code_id}")
    print("  ...")
    
    print("\nPhenotype labels:")
    for label_id, label_name in list(metadata['idToLabel'].items())[:5]:
        print(f"  {label_id}: {label_name}")
    print("  ...")


def inspect_patient(patient, metadata, patient_idx=0):
    """Print detailed information about a patient."""
    print("\n" + "=" * 80)
    print(f"PATIENT {patient_idx} DETAILS")
    print("=" * 80)
    
    print(f"\nNumber of admissions: {len(patient['covars'])}")
    
    # First admission details
    print("\n--- First Admission ---")
    
    print("\nCovariates:")
    covar_ids, covar_values = patient['covars'][0]
    id2var = {v: k for k, v in metadata['var2id'].items()}
    for var_id, value in zip(covar_ids, covar_values):
        var_name = id2var.get(var_id, f"Unknown_{var_id}")
        print(f"  {var_name}: {value}")
    
    print(f"\nNumber of codes: {len(patient['codes'][0])}")
    print(f"Sample codes: {patient['codes'][0][:5]}")
    
    print(f"\nNumber of time series observations: {len(patient['ts'][0])}")
    if len(patient['ts'][0]) > 0:
        print("\nFirst time series observation:")
        var_ids, values, time_gap = patient['ts'][0][0]
        print(f"  Variables: {var_ids[:5]}...")
        print(f"  Values: {values[:5]}...")
        print(f"  Time gap: {time_gap}")
    
    print(f"\nLabels:")
    print(f"  In-hospital mortality: {patient['labels_ihm'][0]}")
    print(f"  Phenotypes: {patient['labels_phe'][0][:5]}...")
    print(f"  Horizons: {patient['horizons'][0]}")


def calculate_statistics(data):
    """Calculate dataset statistics."""
    print("\n" + "=" * 80)
    print("DATASET STATISTICS")
    print("=" * 80)
    
    total_patients = len(data)
    total_admissions = sum(len(p['covars']) for p in data)
    
    print(f"\nTotal patients: {total_patients}")
    print(f"Total admissions: {total_admissions}")
    print(f"Average admissions per patient: {total_admissions / total_patients:.2f}")
    
    # IHM statistics
    ihm_labels = [label for p in data for label in p['labels_ihm']]
    ihm_positive = sum(ihm_labels)
    print(f"\nIn-hospital mortality:")
    print(f"  Positive: {ihm_positive} ({ihm_positive/len(ihm_labels)*100:.2f}%)")
    print(f"  Negative: {len(ihm_labels) - ihm_positive} ({(1-ihm_positive/len(ihm_labels))*100:.2f}%)")
    
    # Phenotype statistics
    phe_labels = np.array([label for p in data for label in p['labels_phe']])
    print(f"\nPhenotypes:")
    print(f"  Total phenotype labels: {phe_labels.shape}")
    print(f"  Average phenotypes per admission: {phe_labels.sum(axis=1).mean():.2f}")
    
    # Time series statistics
    ts_lengths = [len(ts_list) for p in data for ts_list in p['ts']]
    print(f"\nTime series:")
    print(f"  Average observations per admission: {np.mean(ts_lengths):.2f}")
    print(f"  Min observations: {np.min(ts_lengths)}")
    print(f"  Max observations: {np.max(ts_lengths)}")
    print(f"  Median observations: {np.median(ts_lengths):.2f}")
    
    # Code statistics
    code_counts = [len(codes) for p in data for codes in p['codes']]
    print(f"\nCodes:")
    print(f"  Average codes per admission: {np.mean(code_counts):.2f}")
    print(f"  Min codes: {np.min(code_counts)}")
    print(f"  Max codes: {np.max(code_counts)}")


def main():
    """Main function to demonstrate data loading and inspection."""
    # Configuration
    path_data = "data/processed/mimic3-v2"
    bin_type = "uniform"
    
    print("=" * 80)
    print("MIMIC-III Processed Data Inspection")
    print("=" * 80)
    
    # Load metadata
    metadata = load_metadata(path_data, bin_type)
    inspect_metadata(metadata)
    
    # Load training data
    train_data = load_dataset(path_data, 'train', bin_type)
    
    # Inspect first patient
    if len(train_data) > 0:
        inspect_patient(train_data[0], metadata, patient_idx=0)
    
    # Calculate statistics
    calculate_statistics(train_data)
    
    print("\n" + "=" * 80)
    print("✓ Data inspection complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
