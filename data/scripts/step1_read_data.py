"""
Script 1: Read data from MIMIC-III benchmarks and save as trainDataset.pkl

This script:
1. Reads ICD codes (diagnosis and procedures) from MIMIC-III
2. Filters codes based on minimum frequency threshold
3. Re-structures data from mimic3-benchmarks
4. Saves processed data as {split}Dataset.pkl files
"""

import os
import pickle
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm


def expand_path(path: str) -> str:
    """Expand user home directory in paths."""
    return os.path.expanduser(path)


def load_icd_codes(cfg: DictConfig):
    """Load and filter ICD diagnosis and procedure codes."""
    mimic_path = expand_path(cfg.mimic_path)
    min_th_icd = cfg.min_th_icd
    path_data = cfg.path_data
    
    print(f"Loading ICD codes from {mimic_path}")
    
    # Load diagnosis ICD codes
    df_icd = pd.read_csv(
        os.path.join(mimic_path, "DIAGNOSES_ICD.csv")
    ).sort_values(['SUBJECT_ID', 'HADM_ID', 'SEQ_NUM'])
    
    freqs = df_icd['ICD9_CODE'].value_counts()
    freqs = freqs[freqs > min_th_icd]
    print(f"Number of diagnosis ICD codes after filtering: {len(freqs)}")
    
    if cfg.save_plots:
        fig = px.bar(freqs.head(10), title='Top 10 ICD9 codes (Diagnosis)')
        os.makedirs(f"{path_data}/plots", exist_ok=True)
        fig.write_html(f"{path_data}/plots/icd_diagnosis_top10.html")
    
    df_icd = df_icd[df_icd['ICD9_CODE'].isin(freqs.index)]
    
    # Load procedure ICD codes
    df_proc = pd.read_csv(
        os.path.join(mimic_path, "PROCEDURES_ICD.csv")
    ).sort_values(['SUBJECT_ID', 'HADM_ID', 'SEQ_NUM'])
    
    df_proc['ICD9_CODE'] = df_proc['ICD9_CODE'].astype(str)
    
    freqs_proc = df_proc['ICD9_CODE'].value_counts()
    freqs_proc = freqs_proc[freqs_proc > min_th_icd]
    print(f"Number of procedure ICD codes after filtering: {len(freqs_proc)}")
    
    if cfg.save_plots:
        fig = px.bar(freqs_proc.head(10), title='Top 10 ICD9 codes (Procedures)')
        fig.write_html(f"{path_data}/plots/icd_procedures_top10.html")
    
    df_proc = df_proc[df_proc['ICD9_CODE'].isin(freqs_proc.index)]
    
    return df_icd, df_proc


def process_benchmarks_data(cfg: DictConfig, df_icd: pd.DataFrame, df_proc: pd.DataFrame):
    """Process data from MIMIC-III benchmarks and save as pickle files."""
    path_root = expand_path(cfg.path_root)
    path_phe = expand_path(cfg.path_phe)
    path_data = cfg.path_data
    
    ts_vars = cfg.ts_cat + cfg.ts_cont
    ts_hours = cfg.ts_hours
    
    os.makedirs(path_data, exist_ok=True)
    
    for split in cfg.splits:
        print(f"\nProcessing {split} split...")
        df_split = pd.read_csv(f"{path_phe}/{split}_listfile.csv")
        
        data = []
        list_sids = []
        
        for _, row in tqdm(df_split.iterrows(), total=df_split.shape[0], desc=f"Processing {split}"):
            stay = row.stay
            sid = int(stay.split('_')[0])
            order = int(stay.split('_')[1][7:]) - 1  # admission id, 0-indexed
            label_phe = row[2:].values.astype(int)
            
            # Getting hadm_id and label_ihm from root directory
            ff = 'test' if split == 'test' else 'train'
            stay_csv = pd.read_csv(f"{path_root}/{ff}/{sid}/stays.csv")
            
            hadm_id = stay_csv.iloc[order].HADM_ID
            label_ihm = stay_csv.iloc[order].MORTALITY_INHOSPITAL
            
            # Covariates: [Age, Gender]
            covariates = [
                stay_csv.iloc[order].AGE,
                1 if stay_csv.iloc[order].GENDER == 'M' else 0,
            ]
            
            # ICD and procedure codes
            codes_icd = ['icd_' + x for x in df_icd[df_icd.HADM_ID == hadm_id]['ICD9_CODE'].tolist()]
            codes_proc = ['proc_' + x for x in df_proc[df_proc.HADM_ID == hadm_id]['ICD9_CODE'].tolist()]
            codes = [codes_icd, codes_proc]
            
            # Time series data
            df_ts = pd.read_csv(f"{path_phe}/{ff}/{stay}")[ts_vars + ts_hours]
            
            # Limit precision: hours to integer, other variables to 2 decimal places
            for col in ts_hours:
                if col in df_ts.columns:
                    df_ts[col] = df_ts[col].round(0).astype('Int64')  # Use Int64 to handle NaN
            for col in ts_vars:
                if col in df_ts.columns and pd.api.types.is_numeric_dtype(df_ts[col]):
                    df_ts[col] = df_ts[col].round(2)
            
            if sid in list_sids:  # If the subject is already in the list
                subject_data = data[list_sids.index(sid)]
                subject_data['hadm_id'].append(hadm_id)
                subject_data['covariates'].append(covariates)
                subject_data['codes'].append(codes)
                subject_data['ts'].append(df_ts)
                subject_data['label_ihm'].append(label_ihm)
                subject_data['label_phe'].append(label_phe)
            else:  # If the subject is not in the list
                subject_data = {
                    'sid': sid,  # subject id
                    'hadm_id': [hadm_id],  # admission id
                    'covariates': [covariates],  # covariates
                    'codes': [codes],  # icd and proc codes
                    'ts': [df_ts],  # time series data
                    'label_ihm': [label_ihm],  # ihm label
                    'label_phe': [label_phe]  # phe label
                }
                list_sids.append(sid)
                data.append(subject_data)
        
        # Save to pickle
        print(f"Saving {split} dataset with {len(data)} patients...")
        with open(f"{path_data}/{split}Dataset.pkl", "wb") as f:
            pickle.dump(data, f)
        print(f"✓ Saved {split}Dataset.pkl")


@hydra.main(version_base=None, config_path="../../configs/data", config_name="prepare_mimic")
def main(cfg: DictConfig):
    """Main function to process MIMIC-III data."""
    print("=" * 80)
    print("STEP 1: Reading data from MIMIC-III benchmarks")
    print("=" * 80)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    
    # Load ICD codes
    df_icd, df_proc = load_icd_codes(cfg)
    
    # Process benchmarks data
    process_benchmarks_data(cfg, df_icd, df_proc)
    
    print("\n" + "=" * 80)
    print("✓ STEP 1 COMPLETED: Data successfully processed and saved")
    print("=" * 80)


if __name__ == "__main__":
    main()
