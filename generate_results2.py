#!/usr/bin/env python
"""
Generate comprehensive evaluation results for synthetic EHR data.

This script evaluates synthetic EHR data across three dimensions:
- Fidelity: N-gram analysis, correlation matrices, PRDC metrics
- Utility: Downstream task performance (mortality prediction, phenotyping)
- Privacy: Distance-based metrics for both ICD codes and time series
"""
import torch
import os
import pickle
import random
from collections import defaultdict
from pathlib import Path

import dill
import hydra
import numpy as np
import pandas as pd
import wandb
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

# Visualization
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
from sklearn.metrics import ConfusionMatrixDisplay, roc_auc_score

# Statistical imports
from scipy.stats import pearsonr, gaussian_kde, wasserstein_distance
from scipy.spatial.distance import jensenshannon

# OpenTSNE for dimensionality reduction
from openTSNE import TSNE
from datasets import load_from_disk
# Project imports
from synehrgy.utils import (
    genTSembeddings,
    plot_corr3,
    compute_synthcity2,
    compute_utility2,
    compute_mia_knn,
    compute_nnaa,
)

from synehrgy.Dataset import ClinicalDataset
from synehrgy.models import SynEHRgy

# ==================== HELPER FUNCTIONS ====================

def sample_from_discretization(i, name, discretization):
    """Uniform sampling from discretized intervals."""
    return random.uniform(discretization[name][i], discretization[name][i + 1])


def get_df_ts_covars(k, dataset, metadata, path_ts_cache, force_compute=False, save=True):
    """
    Convert discretized time series data to DataFrame with continuous and categorical values.
    
    Args:
        k: Dataset name/key
        dataset: Input dataset
        metadata: Metadata dictionary
        path_ts_cache: Cache directory path
        force_compute: Force recomputation even if cached
        save: Save results to disk
        
    Returns:
        (dfs_ts, dfs_covar): Time series and covariate DataFrames
    """
    cache_ts = f"{path_ts_cache}/df-ts-{k}.pkl"
    cache_static = f"{path_ts_cache}/df-static-{k}.pkl"
    
    # Extract metadata
    possibleValues = metadata['possibleValues']
    isCategorical = metadata['isCategorical']
    discretization = metadata['discretization']
    var2id = metadata['var2id']
    id2var = {v: k for k, v in var2id.items()}
    ts_info = metadata['ts_info']
    n_phe_labels = 25  # Fixed for MIMIC-III
    
    # Try to load from cache
    if os.path.exists(cache_ts) and not force_compute:
        print(f"Loading cached data for {k}")
        with open(cache_ts, 'rb') as f:
            dfs_ts = pickle.load(f)
        with open(cache_static, 'rb') as f:
            dfs_covar = pickle.load(f)
        return dfs_ts, dfs_covar
    
    print(f"Processing time series data for {k}...")
    sub_dataset = dataset[:]
    all_dfs = []
    all_covars = []
    
    for i_p, patient in tqdm(enumerate(sub_dataset), total=len(sub_dataset), desc=k):
        # Convert covariates
        if len(patient['covars']) == 0:
            continue
            
        covars = patient['covars'][0]
        labels = patient['labels_phe'][0]
        temp_covar = {'id': i_p}
        
        # Add labels
        for i, label in enumerate(labels):
            temp_covar[f'label_phe_{i}'] = label
        temp_covar['label_ihm'] = patient['labels_ihm'][0]
        
        # Add covariate values
        if len(covars) > 0:
            for covar_id, covar_value in zip(covars[0], covars[1]):
                name = id2var[covar_id]
                if isCategorical[name]:
                    covar_value = possibleValues[name][covar_value]
                else:
                    covar_value = sample_from_discretization(covar_value, name, discretization)
                temp_covar[name] = covar_value
        
        all_covars.append(temp_covar)
        
        # Process time series (first admission only)
        for admission in patient['ts'][:1]:
            prev_time = 0
            ts_data = []
            
            for measurement in admission:
                if measurement[1] == []:
                    continue
                
                indices = measurement[0]
                values = measurement[1]
                time_gap = measurement[2][0]
                
                timestamp = prev_time + sample_from_discretization(time_gap, 'Hours', discretization)
                prev_time = timestamp
                
                empty_rec = {'id': i_p, 'Hours': timestamp}
                empty_rec.update({name: np.nan for name in list(ts_info.keys())})
                
                for idx, value in zip(indices, values):
                    name = id2var[idx]
                    if isCategorical[name]:
                        empty_rec[name] = value
                    else:
                        try:
                            empty_rec[name] = sample_from_discretization(value, name, discretization)
                        except Exception as e:
                            print(f"Error processing {name}, {value}: {e}")
                
                ts_data.append(empty_rec)
            
            all_dfs.append(pd.DataFrame(ts_data))
    
    # Combine dataframes
    dfs_ts = pd.concat(all_dfs)
    dfs_covar = pd.DataFrame(all_covars)
    
    # Keep only common IDs
    print(f"Before filtering: {dfs_ts.id.nunique()} TS patients, {dfs_covar.id.nunique()} static patients")
    common_ids = pd.merge(dfs_ts[['id']], dfs_covar[['id']], on='id')
    dfs_ts = dfs_ts[dfs_ts['id'].isin(common_ids['id'])]
    dfs_covar = dfs_covar[dfs_covar['id'].isin(common_ids['id'])]
    print(f"After filtering: {dfs_ts.id.nunique()} TS patients, {dfs_covar.id.nunique()} static patients")
    
    # Save to cache
    if save:
        os.makedirs(path_ts_cache, exist_ok=True)
        with open(cache_ts, 'wb') as f:
            pickle.dump(dfs_ts, f)
        with open(cache_static, 'wb') as f:
            pickle.dump(dfs_covar, f)
        print(f"Cached data saved for {k}")
    
    return dfs_ts, dfs_covar


def compute_ngram_dict(sequences, n):
    """Compute n-gram counts for sequences."""
    ngram_dict = defaultdict(int)
    
    for seq in sequences:
        for i in range(len(seq) - n + 1):
            ngram = tuple(seq[i:i+n])
            ngram_dict[ngram] += 1
    
    return dict(ngram_dict)


def compute_bigram_seq(sequences):
    """Compute sequential bi-grams across patient visits."""
    ngram_dict = defaultdict(int)
    total_w1 = {}
    
    for seq in sequences:
        sub1, sub2 = seq[0], seq[1]
        for i in range(len(sub1)):
            for j in range(len(sub2)):
                bigram = (sub1[i], sub2[j])
                ngram_dict[bigram] += 1
                total_w1[sub1[i]] = total_w1.get(sub1[i], 0) + 1
    
    # Normalize by first word occurrence
    ngram_dict = {k: v / total_w1[k[0]] for k, v in ngram_dict.items()}
    return dict(ngram_dict)


def compute_jsd(d1, d2):
    """Compute Jensen-Shannon divergence between two distributions using KDE."""
    if len(d1) < 10 or len(d2) < 10:
        return np.nan
    
    kde1 = gaussian_kde(d1)
    kde2 = gaussian_kde(d2)
    
    max_val = max(max(d1), max(d2))
    x = np.linspace(0, max_val, 1000)
    p1 = kde1(x)
    p2 = kde2(x)
    
    return jensenshannon(p1, p2)


def plot_tsne(data: dict, cfg) -> go.Figure:
    """Create t-SNE visualization for multiple datasets."""
    X = []
    N = cfg.tsne.n_samples
    
    for k in data.keys():
        if data[k].shape[0] < N:
            X.append(data[k])
            
        else:
            random_indices = np.random.choice(data[k].shape[0], size=N, replace=False)
            
            X.append(data[k][random_indices, :])
    
    X = np.concatenate(X, axis=0)
    
    tsne = TSNE(
        n_components=cfg.tsne.n_components,
        perplexity=cfg.tsne.perplexity,
        learning_rate=cfg.tsne.learning_rate,
        n_jobs=cfg.tsne.n_jobs
    )
    X_tsne = tsne.fit(X)
    
    fig_tsne = go.Figure()
    
    for k in data.keys():
        fig_tsne.add_trace(
            go.Scatter(x=X_tsne[:N, 0], y=X_tsne[:N, 1], mode="markers", name=k)
        )
        X_tsne = X_tsne[N:]
    
    fig_tsne.update_traces(marker=dict(opacity=0.75, size=5))
    
    return fig_tsne


def hamming_distance(seq1, seq2):
    """Compute set-based distance between two sequences."""
    return len(set(seq1) ^ set(seq2))


def pairwise_hamming_distance(dataset1, dataset2):
    """Compute pairwise Hamming distance matrix between two datasets."""
    len1, len2 = len(dataset1), len(dataset2)
    D = np.zeros((len1, len2), dtype=int)
    
    for i in tqdm(range(len1), desc="Computing distances"):
        for j in range(len2):
            D[i, j] = hamming_distance(dataset1[i], dataset2[j])
    
    return D


def comp_acc(d1, d2):
    """Compute balanced accuracy for distance comparison."""
    acc = 0.5 * (sum(d1 < d2) / len(d1) + sum(d1 > d2) / len(d1))
    return acc


def prettify_metrics(all_metrics):
    """Convert list of metric dictionaries to formatted DataFrame."""
    metric_names = list(pd.DataFrame(all_metrics[0]).index)
    columns = list(pd.DataFrame(all_metrics[0]).columns)
    
    mat = np.stack([pd.DataFrame(m).values for m in all_metrics])
    
    df_mean = pd.DataFrame(np.mean(mat, axis=0), columns=columns, index=metric_names).round(3).T
    df_std = pd.DataFrame(np.std(mat, axis=0), columns=columns, index=metric_names).round(3).T
    
    # Combine mean and std
    df = df_mean.astype(str)
    return df


# ==================== ANALYSIS FUNCTIONS ====================



def analyze_fidelity(X, cfg, output_dir, data_name):
    """Analyze fidelity for time series using correlation and PRDC metrics."""
    print("\n" + "="*50)
    print("FIDELITY ANALYSIS: TIME SERIES")
    print("="*50)
    
    CONT_VARS = list(cfg.continuous_vars)
    
        
    # PRDC metrics
    print("\nComputing PRDC metrics...")
    # LL = 4 * len(CONT_VARS)
    all_metrics = []
    
    for random_state in cfg.random_seeds:
        print(f"  Random state: {random_state}")
        Xy = {}
        
        for k in ['train', 'target']:
            print(f"    Processing {k} dataset...",len(X[k]))
            Xy[k] = X[k]#.fillna(0).iloc[:, :LL]
            # Xy[k] = Xy[k].sample(cfg.n_samples_utility, random_state=random_state, replace=False)
            n = Xy[k].shape[0]
            idx = torch.randperm(n, generator=torch.Generator().manual_seed(random_state))[:cfg.n_samples_utility]
            Xy[k] = Xy[k][idx]
            Xy[k] = Xy[k] + np.random.normal(0, 0.00001, Xy[k].shape)
        
        metrics_synth = {data_name: compute_synthcity2(Xy['train'], Xy['target'])}
        all_metrics.append(metrics_synth)
    
    # Format results
    df = prettify_metrics(all_metrics)
    
    # Remove unwanted columns
    if 'privacy.identifiability_score.score' in df.columns:
        df = df.drop(columns=['privacy.identifiability_score.score'])
    if 'privacy.identifiability_score.score_OC' in df.columns:
        df = df.drop(columns=['privacy.identifiability_score.score_OC'])
    
    # # Add temporal correlation difference
    # df['TCD'] = str(round(corr_tcd[data_name], 3))
    
    # Log to wandb
    if wandb.run is not None:
        wandb_metrics = {}
        
        # Log PRDC metrics
        for col in df.columns:
            if col != 'TCD':
                try:
                    wandb_metrics[f'fidelity/{col}'] = float(df[col].iloc[0])
                except:
                    pass
        
        wandb.log(wandb_metrics)
    
    # Save results
    df.to_csv(f"{output_dir}/fid.csv")
    print(f"\n✓ Time series fidelity results saved to {output_dir}/fid.csv")
    
    return df



def analyze_privacy(X, y, cfg, output_dir, data_name):
    """Analyze privacy for time series using MIA and NNAA metrics."""
    print("\n" + "="*50)
    print("PRIVACY ANALYSIS: TIME SERIES")
    print("="*50)
    
    # CONT_VARS = list(cfg.continuous_vars)
    # LL = 4 * len(CONT_VARS)
    
    all_metrics_mia = []
    all_metrics_nnaa = []
    
    for random_state in tqdm(cfg.random_seeds, desc="Random states"):
        Xy = {}
        
        for k in ['train', 'test', 'target']:
            # Xy[k] = pd.concat([X[k].fillna(0).iloc[:, :LL], y[k]], axis=1)
            # Xy[k] = Xy[k].sample(cfg.n_samples_privacy, random_state=random_state, replace=False)
            # Xy[k] = Xy[k] + np.random.normal(0, 0.00001, Xy[k].shape)

            Xy[k] = X[k]#.fillna(0).iloc[:, :LL]
            # Xy[k] = Xy[k].sample(cfg.n_samples_utility, random_state=random_state, replace=False)
            n = Xy[k].shape[0]
            idx = torch.randperm(n, generator=torch.Generator().manual_seed(random_state))[:cfg.n_samples_utility]
            Xy[k] = Xy[k][idx]
            Xy[k] = Xy[k] + np.random.normal(0, 0.00001, Xy[k].shape)

        REAL = Xy['train'].cpu().numpy()
        FAKE = Xy['target'].cpu().numpy()
        TEST = Xy['test'].cpu().numpy()

        print(REAL.shape, FAKE.shape, TEST.shape)
        metrics_mia = {data_name: compute_mia_knn(REAL, FAKE, TEST)}
        metrics_nnaa = {data_name: compute_nnaa(REAL, FAKE, TEST)}
        
        all_metrics_mia.append(metrics_mia)
        all_metrics_nnaa.append(metrics_nnaa)
    
    # Format and save results
    df_mia = prettify_metrics(all_metrics_mia)
    df_mia.to_csv(f"{output_dir}/privacy-ts-mia.csv")
    
    df_nnaa = prettify_metrics(all_metrics_nnaa)
    df_nnaa.to_csv(f"{output_dir}/privacy-ts-nnaa.csv")
    
    # Log to wandb
    if wandb.run is not None:
        wandb_metrics = {}
        
        # Log MIA metrics
        for col in df_mia.columns:
            try:
                wandb_metrics[f'privacy/MIA/{col}'] = float(df_mia[col].iloc[0])
            except:
                pass
        
        # Log NNAA metrics
        for col in df_nnaa.columns:
            try:
                wandb_metrics[f'privacy/NNAA/{col}'] = float(df_nnaa[col].iloc[0])
            except:
                pass
        
        wandb.log(wandb_metrics)
    
    print(f"\n✓ Time series privacy results saved to {output_dir}/")
    
    return df_mia, df_nnaa


def create_tsne_visualization(X, cfg, output_dir, data_name):
    """Create t-SNE visualization comparing synthetic and real data."""
    print("\n" + "="*50)
    print("t-SNE VISUALIZATION")
    print("="*50)
    
    # Prepare data for t-SNE
    data_tsne = {}
    for k in ['train', 'test', 'target']:
        if k in X and X[k] is not None:
            data_tsne[k] = X[k]#.fillna(0).values
    
    if 'target' not in data_tsne:
        print("  Warning: No target data for t-SNE visualization")
        return None
    
    print(f"  Creating t-SNE with {cfg.tsne.n_samples} samples per dataset...")
    
    # Create t-SNE plot
    fig_tsne = plot_tsne(data_tsne, cfg)
    fig_tsne.update_layout(
        title=f"t-SNE Visualization - {data_name}",
        template="plotly"
    )
    
    # Save plot locally
    plot_path = f"{output_dir}/tsne.html"
    fig_tsne.write_html(plot_path)
    print(f"  ✓ Saved t-SNE plot to {plot_path}")
    
    # Log to wandb
    if wandb.run is not None:
        wandb.log({'visualization/tsne': wandb.Html(fig_tsne.to_html())})
    
    print("✓ t-SNE visualization complete")
    
    return fig_tsne

@hydra.main(version_base=None, config_path="configs", config_name="results")
def main(cfg: DictConfig):
    """Main function to generate all results."""
    print("="*70)
    print("SynEHRgy Results Generation")
    print("="*70)

    
    # Get data name from command line
    data_name = cfg.get('data_name', 'synehrgy-mimic-v2')
    print(f"\nDataset: {data_name}")
    
    # Determine if this is a real split or synthetic data
    is_real_split = data_name in ['train', 'test', 'val']
    
    
    
    # Setup output directory
    output_dir = f'RESULTS/{data_name}'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}\n")
    
    # Initialize wandb
    if cfg.get('wandb', {}).get('project'):
        if cfg.val_only:
            run_name = f"results2-valonly-{data_name}"
        else:
            run_name = f"results2-{data_name}"
        wandb.init(
            project=cfg.wandb.project,
            name=run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=['evaluation', 'results2', data_name]
        )
        print(f"✓ W&B initialized: {cfg.wandb.project}\n")
    else:
        print("⚠ W&B not configured, skipping logging\n")
    
    # # ==================== LOAD METADATA ====================
    # print("Loading metadata...")
    # metadata = pickle.load(open(cfg.paths.meta, 'rb'))
    
    # indexToCode = metadata['idToCode']
    # codeToIndex = metadata['codeToId']
    # var2id = metadata['var2id']
    # id2var = {v: k for k, v in var2id.items()}
    
    # print("✓ Metadata loaded\n")
    
    # ==================== LOAD DATASETS ====================
    # config_path = f"./saved_models/{RUN_NAME}_config.yaml"
    model_path = f"./saved_models/{data_name}"
    syn_folder = "./data/synthetic"
    
    # data_syn = pickle.load(open(f"{syn_folder}/{data_name}Dataset.pkl", 'rb'))
    # data_syn = load_from_disk(f"{syn_folder}/hf_{data_name}Dataset")
    
    datasets = {}
    config_main = OmegaConf.load(f"{model_path}/config_main.yaml")
    print("Loading Clinical datasets...")
    if cfg.val_only:
        # config_main = OmegaConf.load(f"./saved_models/v8-gpt3-var+quant/config_main.yaml")

        datasets['target'] = ClinicalDataset(config_main.data.path, config_main.n_ctx, split='val', disc_name=config_main.disc_name)
    else:
        datasets['target'] = ClinicalDataset(config_main.data.path, config_main.n_ctx, split='synthetic', disc_name=config_main.disc_name, data_name=data_name)
    
    # Load training and test sets
    datasets['train'] = ClinicalDataset(config_main.data.path, config_main.n_ctx, split='train', disc_name=config_main.disc_name)
    datasets['test'] = ClinicalDataset(config_main.data.path, config_main.n_ctx, split='test', disc_name=config_main.disc_name)
    

    for k in datasets.keys():
        datasets[k].discretize(tok_strategy=config_main.tok_strategy, disc_name=config_main.disc_name)
    # datasets['train'].discretize(tok_strategy=config_main.tok_strategy, disc_name=config_main.disc_name)
    # datasets['test'].discretize(tok_strategy=config_main.tok_strategy, disc_name=config_main.disc_name)
    # datasets['target'].discretize(tok_strategy=config_main.tok_strategy, disc_name=config_main.disc_name)

    # loading the model
    trainer = SynEHRgy.from_pretrained(model_path,
                                       train_dataset=datasets['train'],
                                       eval_dataset=datasets['train'],
                                       )

    fig_dist, fig_usage = datasets['train'].plot_token_dist(sample_size=1000)
    
    wandb.log({'visualization/n_token_dist': wandb.Html(fig_dist.to_html())})
    wandb.log({'visualization/n_token_usage': wandb.Html(fig_usage.to_html())})

    # ==================== CREATE TIME SERIES EMBEDDINGS ====================
    print("Compute the embedding")
    X, y = {}, {}
    for k in datasets.keys():
        all_embeddings = trainer.compute_embeddings(datasets[k])
        X[k] = all_embeddings[:,:]

    print("Saving the embedding",all_embeddings.shape)
    
    
    
    
    # ==================== RUN ANALYSES ====================
    analyze_fidelity(X, cfg, output_dir, data_name)
    analyze_privacy(X, y, cfg, output_dir, data_name)
    create_tsne_visualization(X, cfg, output_dir, data_name)
    


    # Finish wandb run
    if wandb.run is not None:
        wandb.finish()
        print("\n✓ W&B run finished")


if __name__ == "__main__":
    main()
