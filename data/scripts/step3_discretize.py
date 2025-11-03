"""
Script 3: Discretize data and save as DiscDataset.pkl

This script:
1. Loads the processed dataset and metadata
2. Discretizes time series data using the token dictionary
3. Converts all data to token IDs
4. Saves discretized data as {split}DiscDataset.pkl files
"""

import os
import pickle
from pathlib import Path
from multiprocessing import Pool, cpu_count

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm


def expand_path(path: str) -> str:
    """Expand user home directory in paths."""
    return os.path.expanduser(path)


def get_index(mapping: dict, key: str, value: float) -> int:
    """
    Get the index of the value in the discretization mapping[key].
    
    Args:
        mapping: Discretization mapping dictionary
        key: Variable name
        value: Value to discretize
    
    Returns:
        Index of the bin
    """
    possible_values = mapping[key]
    for i in range(len(possible_values) - 1):
        if value <= possible_values[i + 1]:
            return i
    if value > possible_values[-1]:
        return len(possible_values) - 2
    return int(len(possible_values) - 2)


def discretize_covariates(covariates: list, covars_list: list, var2id: dict,
                         possibleValues: dict, isCategorical: dict, 
                         discretization: dict) -> tuple:
    """
    Discretize covariate values.
    
    Args:
        covariates: List of covariate values
        covars_list: List of covariate names
        var2id: Variable to ID mapping
        possibleValues: Possible values for categorical variables
        isCategorical: Dictionary indicating if variable is categorical
        discretization: Discretization bins for continuous variables
    
    Returns:
        Tuple of (variable_ids, discretized_values)
    """
    x = []
    y = []
    for var in covars_list:
        x.append(var2id[var])
        if isCategorical[var]:
            y.append(possibleValues[var][covariates[covars_list.index(var)]])
        else:
            y.append(get_index(discretization, var, covariates[covars_list.index(var)]))
    
    return (x, y)


def discretize_codes(codes: list, codeToId: dict) -> list:
    """
    Convert code strings to token IDs.
    
    Args:
        codes: List of [icd_codes, proc_codes]
        codeToId: Code to ID mapping
    
    Returns:
        List of code IDs
    """
    new_code = [codeToId[code] for code in (codes[0] + codes[1])]
    return new_code


def discretize_timeseries(df_ts: pd.DataFrame, var2id: dict, possibleValues: dict,
                         isCategorical: dict, discretization: dict) -> list:
    """
    Discretize time series data.
    
    Args:
        df_ts: Time series dataframe indexed by hours
        var2id: Variable to ID mapping
        possibleValues: Possible values for categorical variables
        isCategorical: Dictionary indicating if variable is categorical
        discretization: Discretization bins for continuous variables
    
    Returns:
        List of tuples (variable_ids, discretized_values, time_gap)
    """
    adm_ts = []
    prev_time = 0
    bad_data = 0
    
    for time, mes in df_ts.iterrows():
        mes = {k: v for k, v in mes.items() if not pd.isnull(v)}
        
        new_labs = []
        new_values = []
        for var, val in mes.items():
            if isCategorical[var]:
                new_labs.append(var2id[var])
                try:
                    new_values.append(possibleValues[var][str(val)])
                except Exception as e:
                    print(f"Error Categorical: {var} {val}")
                    bad_data += 1
            else:  # continuous
                try:
                    new_values.append(get_index(discretization, var, float(val)))
                    new_labs.append(var2id[var])
                except Exception as e:
                    print(f"Error Cont: {var} {val}")
                    bad_data += 1
        
        time_gap = get_index(discretization, "Hours", time - prev_time)
        prev_time = time
        
        if len(new_labs) == len(new_values) and len(new_labs) > 0:
            adm_ts.append((new_labs, new_values, [time_gap]))
    
    return adm_ts, bad_data


def calculate_horizons(hours: list, horizon_values: list) -> list:
    """
    Calculate the index of the maximum hour before each horizon.
    
    Args:
        hours: List of hour timestamps
        horizon_values: List of horizon values to check
    
    Returns:
        List of indices for each horizon
    """
    list_horizons = []
    for horizon in horizon_values:
        try:
            max_hour = max([x for x in hours if x < horizon])
            max_hour_id = hours.index(max_hour)
        except Exception as e:
            print(f"Error: {horizon} {hours}")
            max_hour_id = -1
        list_horizons.append(max_hour_id)
    return list_horizons


def discretize_patient(args):
    """
    Worker function to discretize a single patient's data.
    
    Args:
        args: Tuple of (patient_data, covars_list, horizons, var2id, codeToId, 
              possibleValues, isCategorical, discretization)
    
    Returns:
        Tuple of (discretized_patient, bad_data_count)
    """
    p, covars_list, horizons, var2id, codeToId, possibleValues, isCategorical, discretization = args
    
    all_covars = []
    all_codes = []
    all_ts = []
    all_horizons = []
    patient_bad_data = 0
    
    for i_stay in range(len(p['hadm_id'])):
        # Discretize covariates
        covariates = p['covariates'][i_stay]
        disc_covars = discretize_covariates(
            covariates, covars_list, var2id, possibleValues, 
            isCategorical, discretization
        )
        all_covars.append(disc_covars)
        
        # Discretize codes
        if p['codes'] is not None:
            codes = p['codes'][i_stay]
            disc_codes = discretize_codes(codes, codeToId)
            all_codes.append(disc_codes)
        else:
            all_codes.append([])
        
        # Discretize time series
        df_ts = p['ts'][i_stay].set_index('Hours')
        # remove patientunitstayid column if exists
        if 'patientunitstayid' in df_ts.columns:
            df_ts = df_ts.drop(columns=['patientunitstayid'])
        hours = df_ts.index.tolist()
        
        # Calculate horizons
        list_horizons = calculate_horizons(hours, horizons)
        all_horizons.append(list_horizons)
        
        # Discretize time series data
        adm_ts, bad_data = discretize_timeseries(
            df_ts, var2id, possibleValues, isCategorical, discretization
        )
        all_ts.append(adm_ts)
        patient_bad_data += bad_data
    
    # Create discretized patient record
    disc_patient = {
        'covars': all_covars,
        'codes': all_codes,
        'ts': all_ts,
        'horizons': all_horizons
    }
    if 'label_phe' in p:
        disc_patient.update({
            'labels_phe': p['label_phe'],          
        })
    if 'label_ihm' in p:
        disc_patient.update({
            'labels_ihm': p['label_ihm'],
        })
    if 'label_mortality_48h' in p:
        disc_patient.update({
            'labels_ihm': p['label_mortality_48h'],
        })
    return disc_patient, patient_bad_data


def discretize_dataset(cfg: DictConfig, split: str, metadata: dict) -> list:
    """
    Discretize a complete dataset split.
    
    Args:
        cfg: Configuration
        split: Split name ('train', 'val', or 'test')
        metadata: Metadata dictionary with tokenization info
    
    Returns:
        List of discretized patient records
    """
    path_data = cfg.path_data
    
    # Load raw dataset
    print(f"Loading {split} dataset...")
    with open(f"{path_data}/{split}Dataset.pkl", "rb") as f:
        data = pickle.load(f)
    
    # Extract metadata
    var2id = metadata['var2id']
    codeToId = metadata['codeToId']
    possibleValues = metadata['possibleValues']
    isCategorical = metadata['isCategorical']
    discretization = metadata['discretization']
    
    covars_list = list(cfg.covars_cont) + list(cfg.covars_cat)
    horizons = cfg.horizons
    
    # Determine number of workers
    n_workers = getattr(cfg, 'n_workers', cpu_count())
    print(f"Using {n_workers} workers for parallelization")
    
    # Prepare arguments for each patient
    worker_args = [
        (p, covars_list, horizons, var2id, codeToId, possibleValues, isCategorical, discretization)
        for p in data
    ]
    
    disc_data = []
    total_bad_data = 0
    
    # Process in parallel
    with Pool(processes=n_workers) as pool:
        results = list(tqdm(
            pool.imap(discretize_patient, worker_args),
            total=len(data),
            desc=f"Discretizing {split}"
        ))
    
    # Combine results
    for disc_patient, bad_data in results:
        disc_data.append(disc_patient)
        total_bad_data += bad_data
    
    if total_bad_data > 0:
        print(f"Warning: {total_bad_data} bad data points encountered in {split}")
    
    return disc_data


@hydra.main(version_base=None, config_path="../../configs/data", config_name="prepare_mimic")
def main(cfg: DictConfig):
    """Main function to discretize data."""
    print("=" * 80)
    print("STEP 3: Discretizing data")
    print("=" * 80)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    
    path_data = cfg.path_data
    
    # Load metadata
    metadata_file = f"{path_data}/metadata_{cfg.disc_name}.pkl"
    print(f"\nLoading metadata from {metadata_file}...")
    
    with open(metadata_file, "rb") as f:
        metadata = pickle.load(f)
    
    print(f"✓ Loaded metadata with {len(metadata['token2id'])} tokens")
    
    # Process each split
    for split in cfg.splits:
        print(f"\n{'=' * 80}")
        print(f"Processing {split} split")
        print('=' * 80)
        
        disc_data = discretize_dataset(cfg, split, metadata)
        
        # Save discretized data
        output_file = f"{path_data}/{split}DiscDataset_{cfg.disc_name}.pkl"
        
        with open(output_file, "wb") as f:
            pickle.dump(disc_data, f)
        

        
        print(f"✓ Saved {len(disc_data)} patients to {output_file}")


        # save as HF dataset
        from datasets import Dataset

        hf_dataset = Dataset.from_list(disc_data)
        hf_dataset.save_to_disk(f"{path_data}/hf_{split}DiscDataset_{cfg.disc_name}")
    
    print("\n" + "=" * 80)
    print("✓ STEP 3 COMPLETED: All data discretized and saved")
    print("=" * 80)


if __name__ == "__main__":
    main()
