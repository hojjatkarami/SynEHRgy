"""
Script 2: Create token dictionary and save in metadata.pkl

This script:
1. Creates tokens for ICD codes
2. Tokenizes time series variables (categorical and continuous)
3. Adds covariate, label, and special tokens
4. Saves all metadata including token2id, var2id, discretization info
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
from scipy.ndimage import convolve1d
from tqdm import tqdm


def expand_path(path: str) -> str:
    """Expand user home directory in paths."""
    return os.path.expanduser(path)


def soft_label_with_gaussian(i: int, N: int, kernel_size: int, sigma: float = 1.0):
    """
    Generate a soft label vector with Gaussian smoothing.
    
    Args:
        i: Index for one-hot encoding
        N: Length of the one-hot encoded vector
        kernel_size: Size of the Gaussian kernel (should be odd)
        sigma: Standard deviation of the Gaussian kernel
    
    Returns:
        Soft label vector of length N
    """
    # One-Hot Encoding
    one_hot = np.zeros(N)
    one_hot[i] = 1
    
    # Create Gaussian Kernel
    n_neighbour = (kernel_size - 1) // 2
    x = np.linspace(-n_neighbour, n_neighbour, kernel_size)
    gaussian_kernel = np.exp(-0.5 * (x / sigma) ** 2)
    gaussian_kernel /= gaussian_kernel.sum()
    
    # Convolution with Gaussian Kernel
    soft_label = convolve1d(one_hot, gaussian_kernel, mode='constant')
    
    # Normalize
    soft_label = soft_label / soft_label.sum()
    return soft_label


def find_possible_values(all_values, var):
    """Map categorical values to consistent token indices."""
    vals = all_values.unique().tolist()
    dict_map = None
    tokens = []
    
    if var == 'Glascow coma scale eye opening':
        dict_map = {
            '1 No Response': 0, '3 To speech': 1, '4 Spontaneously': 2, '2 To pain': 3,
            'To Speech': 1, 'Spontaneously': 2, 'To Pain': 3
        }
        tokens = ['No Response', 'To Speech', 'Spontaneously', 'To Pain']
        tokens = ["GCS-eo-" + x for x in tokens]
    elif var == 'Glascow coma scale motor response':
        dict_map = {
            '5 Localizes Pain': 0, '6 Obeys Commands': 1, '4 Flex-withdraws': 2,
            '1 No Response': 3, '2 Abnorm extensn': 4, '3 Abnorm flexion': 5,
            'Abnormal extension': 4, 'Obeys Commands': 1, 'Localizes Pain': 0,
            'No response': 3, 'Flex-withdraws': 2, 'Abnormal Flexion': 5
        }
        tokens = ['Localizes Pain', 'Obeys Commands', 'Flex-withdraws', 
                  'No Response', 'Abnormal extension', 'Abnormal Flexion']
        tokens = ["GCS-mr-" + x for x in tokens]
    elif var == 'Glascow coma scale verbal response':
        dict_map = {
            '1.0 ET/Trach': 0, '5 Oriented': 1, '4 Confused': 2, '1 No Response': 3,
            '2 Incomp sounds': 4, '3 Inapprop words': 5, 'No Response-ETT': 0,
            'Oriented': 1, 'Inappropriate Words': 5, 'Confused': 2,
            'Incomprehensible sounds': 4, 'No Response': 3
        }
        tokens = ['ET/Trach', 'Oriented', 'Confused', 'No Response', 'Incomp sounds', 'Inapprop words']
        tokens = ["GCS-vr-" + x for x in tokens]
    else:
        dict_map = {val: i for i, val in enumerate(vals)}
        tokens = [f"{var}-{x}" for x in vals]
    
    return dict_map, tokens


def load_icd_codes(cfg: DictConfig):
    """Load filtered ICD codes."""
    mimic_path = expand_path(cfg.mimic_path)
    min_th_icd = cfg.min_th_icd
    
    # Load diagnosis ICD codes
    df_icd = pd.read_csv(
        os.path.join(mimic_path, "DIAGNOSES_ICD.csv")
    ).sort_values(['SUBJECT_ID', 'HADM_ID', 'SEQ_NUM'])
    
    freqs = df_icd['ICD9_CODE'].value_counts()
    freqs = freqs[freqs > min_th_icd]
    df_icd = df_icd[df_icd['ICD9_CODE'].isin(freqs.index)]
    
    # Load procedure ICD codes
    df_proc = pd.read_csv(
        os.path.join(mimic_path, "PROCEDURES_ICD.csv")
    ).sort_values(['SUBJECT_ID', 'HADM_ID', 'SEQ_NUM'])
    
    df_proc['ICD9_CODE'] = df_proc['ICD9_CODE'].astype(str)
    freqs_proc = df_proc['ICD9_CODE'].value_counts()
    freqs_proc = freqs_proc[freqs_proc > min_th_icd]
    df_proc = df_proc[df_proc['ICD9_CODE'].isin(freqs_proc.index)]
    
    return df_icd, df_proc


def create_code_tokens(df_icd: pd.DataFrame, df_proc: pd.DataFrame):
    """Create token mappings for ICD codes."""
    freqs_icd = df_icd['ICD9_CODE'].value_counts()
    freqs_icd.index = 'icd_' + freqs_icd.index
    
    freqs_proc = df_proc['ICD9_CODE'].value_counts()
    freqs_proc.index = 'proc_' + freqs_proc.index
    
    freqs = pd.concat([freqs_icd, freqs_proc])
    
    codeToId = {code: i for i, code in enumerate(freqs.index)}
    idToCode = {i: code for i, code in enumerate(freqs.index)}
    
    token2id = {('code', i): i for i in range(len(freqs))}
    
    print(f"Created {len(freqs)} code tokens")
    return token2id, codeToId, idToCode


def load_train_timeseries(cfg: DictConfig):
    """Load time series data from training split."""
    path_data = cfg.path_data
    
    print("Loading training data for time series analysis...")
    data = pickle.load(open(f"{path_data}/trainDataset.pkl", "rb"))
    temp = pd.concat([ts for patient in data for ts in patient['ts']])
    
    # Convert categorical variables to string
    for var in cfg.ts_cat:
        temp[var] = temp[var].astype(str)
    
    # Replace 'nan' with np.nan
    temp[temp == 'nan'] = np.nan
    
    # Convert continuous variables to numeric
    for col in tqdm(cfg.ts_cont + cfg.ts_hours, desc="Converting to numeric"):
        temp[col] = pd.to_numeric(temp[col], errors='coerce')
    
    return temp


def tokenize_timeseries(cfg: DictConfig, temp: pd.DataFrame, token2id: dict):
    """Tokenize time series variables."""
    ts_info = {}
    possibleValues = {}
    discretization = {}
    isCategorical = {}
    var2id = {}
    soft_labels = {}
    beginPos = [0]
    
    bin_type = cfg.bin_type
    path_data = cfg.path_data
    os.makedirs(f"{path_data}/ts", exist_ok=True)
    
    all_vars = list(cfg.ts_cat) + list(cfg.ts_cont)
    
    for i, var in enumerate(tqdm(all_vars, desc="Tokenizing time series")):
        all_values = temp[var]
        missing_rate = pd.isnull(all_values).sum() / len(all_values)
        all_values = all_values[all_values.notnull()]
        
        n_unique = all_values.nunique()
        var_type = 'categorical' if var in cfg.ts_cat else 'continuous'
        
        ts_info[var] = {
            'var_type': var_type,
            'missing_rate': missing_rate
        }
        
        if var_type == 'categorical':
            isCategorical[var] = True
            dict_map, tokens = find_possible_values(all_values, var)
            n_tokens = len(tokens)
            
            var2id[var] = len(var2id)
            token2id.update({('ts', var2id[var], i): i + len(token2id) for i in range(n_tokens)})
            beginPos.append(beginPos[-1] + n_tokens)
            possibleValues[var] = dict_map
            
            if cfg.save_plots:
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=all_values, histnorm='probability', 
                                          name=f"{var}-{var_type}", opacity=0.75))
                fig.write_html(f"{path_data}/ts/_{var}.html")
            
            ts_info[var].update({
                'unique_values': tokens,
                'n_tokens': n_tokens
            })
        else:
            isCategorical[var] = False
            all_values2 = all_values
            
            # Calculate statistics
            ts_info[var]['mean'] = all_values2.mean()
            ts_info[var]['std'] = all_values2.std()
            ts_info[var]['min'] = all_values2.min()
            ts_info[var]['max'] = all_values2.max()
            ts_info[var]['0.025'] = all_values2.quantile(0.025)
            ts_info[var]['0.975'] = all_values2.quantile(0.975)
            ts_info[var]['median'] = all_values2.median()
            
            # Handle outliers
            if (ts_info[var]['max'] - ts_info[var]['0.975']) / (ts_info[var]['0.975'] - ts_info[var]['median']) > 3:
                all_values2 = all_values2[all_values2 <= ts_info[var]['0.975']]
            if (ts_info[var]['0.025'] - ts_info[var]['min']) / (ts_info[var]['median'] - ts_info[var]['0.025']) > 3:
                all_values2 = all_values2[all_values2 >= ts_info[var]['0.025']]
            
            ts_info[var]['mean_2'] = all_values2.mean()
            ts_info[var]['std_2'] = all_values2.std()
            ts_info[var]['min_2'] = all_values2.min()
            ts_info[var]['max_2'] = all_values2.max()
            ts_info[var]['0.025_2'] = all_values2.quantile(0.025)
            ts_info[var]['0.975_2'] = all_values2.quantile(0.975)
            ts_info[var]['median_2'] = all_values2.median()
            
            n_bins = min(cfg.n_bins_default, int(n_unique / 5))
            
            if bin_type == 'uniform':
                binned_data, bin_edges = pd.cut(all_values2, bins=n_bins, retbins=True, duplicates='drop')
            elif bin_type == 'quantile':
                binned_data, bin_edges = pd.qcut(all_values2, n_bins, retbins=True, duplicates='drop')
            
            beginPos.append(beginPos[-1] + n_bins)
            discretization[var] = bin_edges.tolist()
            possibleValues[var] = {f"{var}_{i}": i for i in range(n_bins)}
            var2id[var] = len(var2id)
            token2id.update({('ts', var2id[var], i): i + len(token2id) for i in range(n_bins)})
            
            # Create soft labels
            kernel_size = cfg.soft_label_kernel_size
            for i in range(n_bins):
                soft_labels[('ts', var2id[var], i)] = soft_label_with_gaussian(
                    i, n_bins, kernel_size, sigma=cfg.soft_label_sigma
                )
            
            # Plotting
            bin_counts, _ = np.histogram(all_values2, bins=bin_edges)
            total_count = len(all_values2)
            normalized_bin_counts = bin_counts / total_count
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            if cfg.save_plots:
                fig = go.Figure()
                fig.add_trace(go.Bar(x=bin_centers, y=normalized_bin_counts,
                                    width=np.diff(bin_edges) * 1,
                                    name=f"{var}-{var_type}", opacity=0.5))
                fig.add_trace(go.Histogram(x=all_values2, histnorm='probability',
                                          name=f"{var}-{var_type}", opacity=0.75))
                fig.write_html(f"{path_data}/ts/{var}.html")
            
            ts_info[var].update({
                'n_tokens': n_bins,
                'bin_edges': bin_edges.tolist(),
                'bin_counts': bin_counts.tolist(),
                'normalized_bin_counts': normalized_bin_counts.tolist(),
                'bin_centers': bin_centers.tolist(),
                'bin_labels': [f"{var}_{i}" for i in range(n_bins)]
            })
    
    beginPos.pop()  # Remove last element
    
    return token2id, var2id, ts_info, possibleValues, isCategorical, discretization, soft_labels, beginPos


def add_covariate_and_label_tokens(cfg: DictConfig, token2id: dict, var2id: dict, 
                                   possibleValues: dict, isCategorical: dict, 
                                   discretization: dict):
    """Add tokens for covariates, labels, and special tokens."""
    
    # Manual tokenization for covariates
    possibleValues["Gender"] = {0: 0, 1: 1}
    isCategorical["Age"] = False
    isCategorical["Gender"] = True
    isCategorical["Hours"] = False
    
    discretization["Age"] = cfg.age_bins
    discretization["Hours"] = cfg.hours_bins
    
    # Add age tokens
    var2id["Age"] = len(var2id)
    token2id.update({('covar', var2id["Age"], i): i + len(token2id) 
                    for i in range(len(discretization["Age"]) - 1)})
    
    # Add gender tokens
    var2id["Gender"] = len(var2id)
    token2id.update({('covar', var2id["Gender"], i): i + len(token2id) for i in range(2)})
    
    # Add hours tokens
    var2id["Hours"] = len(var2id)
    token2id.update({('timestamp', var2id["Hours"], i): i + len(token2id) 
                    for i in range(len(discretization["Hours"]) - 1)})
    
    # Add label tokens
    token2id.update({('label', 'phe', i): len(token2id) + i for i in range(25)})
    token2id.update({('label', 'ihm', i): len(token2id) + i for i in range(2)})
    
    # Add special tokens
    special_tokens = ['<s>', '</covar>', '</label>', '</code>', '</ts>', '</adm>', 
                     '</s>', '<pad>', '<history>', '<forecast>', '</forecast>']
    for token in special_tokens:
        token2id[token] = len(token2id)
    
    print(f"Total tokens: {len(token2id)}")
    
    return token2id, var2id, possibleValues, isCategorical, discretization


def create_soft_label_matrix(token2id: dict, soft_labels: dict):
    """Create soft label matrix for all tokens."""
    M_soft_labels = np.eye(len(token2id))
    for token in token2id.keys():
        if token in soft_labels:
            temp = soft_labels[token]
            max_pos = np.argmax(temp)
            l_left = token2id[token] - max_pos
            l_right = token2id[token] + len(temp) - max_pos
            M_soft_labels[token2id[token], l_left:l_right] = temp
    
    return M_soft_labels


def extract_phenotype_names(cfg: DictConfig):
    """Extract phenotype label names."""
    path_phe = expand_path(cfg.path_phe)
    phe_names = pd.read_csv(f"{path_phe}/train/listfile.csv").columns[2:].tolist()
    idToLabel = {i: label for i, label in enumerate(phe_names)}
    return idToLabel


def calculate_vocab_size(codeToId: dict, ts_info: dict, discretization: dict, cfg: DictConfig):
    """Calculate vocabulary sizes for different token types."""
    vocab_size = {
        'codes': len(codeToId),
        'lab_cont': sum([ts_info[var]['n_tokens'] for var in cfg.ts_cat]),
        'lab_cat': sum([ts_info[var]['n_tokens'] for var in cfg.ts_cont]),
        'gap': len(discretization['Hours']) - 1,
        'covars': len(discretization['Age']) - 1 + 2,
    }
    return vocab_size


@hydra.main(version_base=None, config_path="../../configs/data", config_name="prepare_mimic")
def main(cfg: DictConfig):
    """Main function to create token dictionary."""
    print("=" * 80)
    print("STEP 2: Creating token dictionary")
    print("=" * 80)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    
    # Load ICD codes and create code tokens
    df_icd, df_proc = load_icd_codes(cfg)
    token2id, codeToId, idToCode = create_code_tokens(df_icd, df_proc)
    
    # Load training time series data
    temp = load_train_timeseries(cfg)
    
    # Tokenize time series
    token2id, var2id, ts_info, possibleValues, isCategorical, discretization, soft_labels, beginPos = \
        tokenize_timeseries(cfg, temp, token2id)
    
    # Add covariate and label tokens
    token2id, var2id, possibleValues, isCategorical, discretization = \
        add_covariate_and_label_tokens(cfg, token2id, var2id, possibleValues, 
                                      isCategorical, discretization)
    
    # Create soft label matrix
    M_soft_labels = create_soft_label_matrix(token2id, soft_labels)
    
    # Extract phenotype names
    idToLabel = extract_phenotype_names(cfg)
    
    # Calculate vocabulary size
    vocab_size = calculate_vocab_size(codeToId, ts_info, discretization, cfg)
    
    print("\nVocabulary sizes:")
    for k, v in vocab_size.items():
        print(f"  {k}: {v}")
    
    # Save metadata
    metadata = {
        'codeToId': codeToId,
        'idToCode': idToCode,
        'ts_info': ts_info,
        'token2id': token2id,
        'var2id': var2id,
        'beginPos': beginPos,
        'possibleValues': possibleValues,
        'isCategorical': isCategorical,
        'discretization': discretization,
        'idToLabel': idToLabel,
        'vocab_size': vocab_size,
        'M_soft_labels': M_soft_labels
    }
    
    path_data = cfg.path_data
    metadata_file = f"{path_data}/metadata2.pkl" if cfg.bin_type == 'uniform' else f"{path_data}/metadata.pkl"
    
    with open(metadata_file, "wb") as f:
        pickle.dump(metadata, f)
    
    print(f"\n✓ Saved metadata to {metadata_file}")
    print("=" * 80)
    print("✓ STEP 2 COMPLETED: Token dictionary created and saved")
    print("=" * 80)


if __name__ == "__main__":
    main()
