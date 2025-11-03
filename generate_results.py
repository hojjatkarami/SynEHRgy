#!/usr/bin/env python
"""
Generate comprehensive evaluation results for synthetic EHR data.

This script evaluates synthetic EHR data across three dimensions:
- Fidelity: N-gram analysis, correlation matrices, PRDC metrics
- Utility: Downstream task performance (mortality prediction, phenotyping)
- Privacy: Distance-based metrics for both ICD codes and time series
"""

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

# Project imports
from synehrgy.utils import (
    genTSembeddings,
    plot_corr3,
    compute_synthcity2,
    compute_utility2,
    compute_mia_knn,
    compute_nnaa,
)


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

def analyze_fidelity_icd(datasets_icd, metadata, output_dir, cfg, data_name):
    """Analyze fidelity for ICD codes using n-gram analysis."""
    if 'target' not in datasets_icd:
        print(f"  Warning: No ICD code dataset found for {data_name}")
        return None

    print("\n" + "="*50)
    print("FIDELITY ANALYSIS: ICD CODES")
    print("="*50)
    
    N_WORDS = metadata['vocab_size']['codes']
    
    # Compute n-grams
    codes_ngram = {}
    codes = {}
    
    for n in range(1, 4):
        print(f"\nComputing {n}-grams...")
        codes_ngram[n] = {}
        codes[n] = {}
        
        for k, dataset in datasets_icd.items():
            temp = [p["visits"][0] for p in dataset]
            codes_ngram[n][k] = compute_ngram_dict(temp, n)
            
            # Compute probabilities
            if n == 1:
                codes[n][k] = {
                    word: count / N_WORDS 
                    for word, count in codes_ngram[n][k].items()
                }
            else:
                codes[n][k] = {
                    comb: count / codes_ngram[n-1][k][comb[:(n-1)]] 
                    for comb, count in codes_ngram[n][k].items()
                }
    
    # Analyze n-gram fidelity
    df_ngram = pd.DataFrame(columns=['1-gram', '2-gram', '3-gram'], index=[data_name])
    wandb_metrics = {}
    
    for n in range(1, 4):
        print(f"\nAnalyzing {n}-grams...")
        
        common_codes = list(set(codes[n]['train'].keys()).intersection(set(codes[n]['target'].keys())))
        common_codes = sorted(common_codes, key=lambda x: codes[n]['train'][x], reverse=True)[:cfg.n_top_ngrams]
        
        data1 = [codes[n]['train'][x] for x in common_codes]
        data2 = [codes[n]['target'][x] for x in common_codes]
        
        metric_val, _ = pearsonr(data1, data2)
        print(f"  Pearson correlation: {metric_val:.3f}")
        df_ngram.loc[data_name, f'{n}-gram'] = metric_val
        wandb_metrics[f'fidelity-icd/{n}-gram'] = metric_val
        
        # Create scatter plot
        fig = go.Figure()
        
        L1 = len(datasets_icd['train'])
        L2 = len(datasets_icd['target'])
        hovertexts = [
            f"{x}: ({int(freq1*L1)}, {int(freq2*L2)})" 
            for x, freq1, freq2 in zip(common_codes, data1, data2)
        ]
        
        maxVal = max(data1)
        minVal = min(data1)
        
        fig.add_trace(go.Scatter(
            x=data1, y=data2, mode='markers', 
            name=f'train-{data_name}', hovertext=hovertexts
        ))
        
        # Add diagonal reference line
        fig.add_trace(go.Scatter(
            x=[minVal, maxVal*1.1], y=[minVal, maxVal*1.1],
            mode="lines", line=dict(color="red"), name="y=x"
        ))
        
        fig.update_layout(
            xaxis_title="Train", 
            yaxis_title=data_name,
            template="plotly", 
            title=f"{n}-gram Analysis (Pearson: {metric_val:.3f})"
        )
        
        # Save plot locally
        plot_path = f"{output_dir}/{n}-gram.html"
        fig.write_html(plot_path)
        print(f"  ✓ Saved plot to {plot_path}")
        
        # Log to wandb
        if wandb.run is not None:
            wandb.log({f'fidelity-icd/{n}-gram-plot': wandb.Html(fig.to_html())})
    
    # Log metrics to wandb
    if wandb.run is not None:
        wandb.log(wandb_metrics)
    
    # Save results
    df_ngram.astype(float).round(3).to_csv(f"{output_dir}/fid-icd-ngram.csv")
    print(f"\n✓ ICD n-gram results saved to {output_dir}/fid-icd-ngram.csv")
    
    return df_ngram


def analyze_fidelity_ts(datasets_df_ts, datasets_df_static, X, y, cfg, output_dir, data_name):
    """Analyze fidelity for time series using correlation and PRDC metrics."""
    print("\n" + "="*50)
    print("FIDELITY ANALYSIS: TIME SERIES")
    print("="*50)
    
    CONT_VARS = list(cfg.continuous_vars)
    
    # Correlation analysis
    print("\nComputing temporal correlations...")
    corr_tcd = {}
    
    print(f"  Processing {data_name}...")
    conf_mat, corr_tcd[data_name] = plot_corr3(
        datasets_df_ts['train'], datasets_df_ts['target'], CONT_VARS, 
        corr_th=0.0, corr_method='ffill'
    )
    print(f"  Temporal Correlation Difference: {corr_tcd[data_name]:.4f}")
    
    # Plot correlation confusion matrix
    LABEL_MAP = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 100: 'nan'}
    SELECTED_LABELS = [0, 1, 2, 3, 4]
    
    conf_normalized = conf_mat
    disp = ConfusionMatrixDisplay(
        confusion_matrix=conf_normalized, 
        display_labels=[LABEL_MAP[i] for i in SELECTED_LABELS]
    )
    fig_corr, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, xticks_rotation=45)
    ax.set_title(f"Temporal Correlation Matrix - {data_name}")
    
    # Save plot locally
    corr_plot_path = f"{output_dir}/corr_{data_name}.png"
    plt.savefig(corr_plot_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved correlation plot to {corr_plot_path}")
    
    # Log to wandb
    if wandb.run is not None:
        wandb.log({
            'fidelity-ts/correlation-matrix': wandb.Image(corr_plot_path),
            'fidelity-ts/TCD': corr_tcd[data_name]
        })
    
    plt.close()
    
    # Co-occurrence matrix (pairwise missingness)
    print("\nComputing co-occurrence matrix...")
    df = datasets_df_ts['target'][CONT_VARS]
    mat = df.notnull().astype(int).values
    c = mat.T @ mat
    x = mat.sum(axis=0)[None,:] + mat.sum(axis=0)[:,None]
    c = np.round(c / x * 100, 2)
    np.fill_diagonal(c, np.nan)
    c[np.triu_indices_from(c, 1)] = np.nan
    
    # Plot co-occurrence matrix
    fig_occ = go.Figure()
    fig_occ.add_trace(go.Heatmap(z=c[::-1], colorscale='Viridis', zmin=0, zmax=50))
    fig_occ.update_layout(
        title_text=f"Co-occurrence Matrix - {data_name}",
        template="plotly",
        height=400,
        width=400,
    )
    
    # Save plot locally
    occ_plot_path = f"{output_dir}/occ-{data_name}.html"
    fig_occ.write_html(occ_plot_path)
    print(f"  ✓ Saved co-occurrence plot to {occ_plot_path}")
    
 
    # Log to wandb
    if wandb.run is not None:
        wandb.log({'fidelity-ts/co-occurrence-matrix': wandb.Html(fig_occ.to_html())})
    
    # PRDC metrics
    print("\nComputing PRDC metrics...")
    LL = 4 * len(CONT_VARS)
    all_metrics = []
    
    for random_state in cfg.random_seeds:
        print(f"  Random state: {random_state}")
        Xy = {}
        
        for k in ['train', 'target']:
            print(f"    Processing {k} dataset...",len(X[k]))
            Xy[k] = X[k].fillna(0).iloc[:, :LL]
            Xy[k] = Xy[k].sample(cfg.n_samples_utility, random_state=random_state, replace=False)
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
    
    # Add temporal correlation difference
    df['TCD'] = str(round(corr_tcd[data_name], 3))
    
    # Log to wandb
    if wandb.run is not None:
        wandb_metrics = {}
        
        # Log PRDC metrics
        for col in df.columns:
            if col != 'TCD':
                try:
                    wandb_metrics[f'fidelity-ts/{col}'] = float(df[col].iloc[0])
                except:
                    pass
        
        wandb.log(wandb_metrics)
    
    # Save results
    df.to_csv(f"{output_dir}/fid-ts.csv")
    print(f"\n✓ Time series fidelity results saved to {output_dir}/fid-ts.csv")
    
    return df


def analyze_utility(X, y, cfg, output_dir, data_name):
    """Analyze utility through downstream task performance."""
    print("\n" + "="*50)
    print("UTILITY ANALYSIS")
    print("="*50)
    
    train_fake_ratios = [tuple(r) for r in cfg.train_fake_ratios]
    
    print("Training XGBoost models for utility evaluation...")
    metrics_utility = compute_utility2(
        X['train'].fillna(0), y['train'],
        X['test'].fillna(0), y['test'],
        X['target'].fillna(0), y['target'],
        train_fake_ratio=train_fake_ratios
    )
    
    # Extract all metrics
    all_metrics = list(metrics_utility[(0, 1)].keys())
    
    # Wandb logging
    wandb_metrics = {}
    
    # Save results for each metric
    for metric in all_metrics:
        dict_metric = {}
        
        # Train only curve
        curve_train_only = {k[0]: v[metric] for k, v in metrics_utility.items() if k in [(0.1, 0), (0.2, 0), (0.5, 0), (1, 0)]}
        dict_metric['train only'] = [0] + list(curve_train_only.values())
        
        # Target dataset curve
        curve_target = {k[0]: v[metric] for k, v in metrics_utility.items() if k in [(0, 1), (0.1, 1), (0.2, 1), (0.5, 1), (1, 1)]}
        dict_metric[data_name] = list(curve_target.values())
        
        # Create utility curve plot
        fig = go.Figure()
        
        # Train only curve
        x_train = list(curve_train_only.keys())
        y_train = list(curve_train_only.values())
        fig.add_trace(go.Scatter(
            x=x_train, y=y_train, mode='lines+markers', 
            name='train only', line=dict(dash='dash', color='black')
        ))
        
        # Target dataset curve
        x_target = list(curve_target.keys())
        y_target = list(curve_target.values())
        fig.add_trace(go.Scatter(
            x=x_target, y=y_target, mode='lines+markers', 
            name=data_name
        ))
        
        fig.update_layout(
            title=f"{metric} - Utility Curve",
            xaxis_title="Fraction of Real Training Data",
            yaxis_title=metric,
            template="plotly"
        )
        
        # Save plot locally
        plot_path = f"{output_dir}/utility-{metric}.html"
        fig.write_html(plot_path)
        
        # Log to wandb
        if wandb.run is not None:
            # Log TSTR (0% real, 100% synthetic)
            wandb_metrics[f'utility/{metric}/TSTR'] = metrics_utility[(0, 1)][metric]
            # Log TRTR (100% real, 0% synthetic)
            wandb_metrics[f'utility/{metric}/TRTR'] = metrics_utility[(1, 0)][metric]
            # Log augmentation (100% real + 100% synthetic)
            wandb_metrics[f'utility/{metric}/augmented'] = metrics_utility[(1, 1)][metric]
            # Log plot
            wandb_metrics[f'utility/{metric}/plot'] = wandb.Html(fig.to_html())
        
        # Save dataframe
        df_metric = pd.DataFrame(dict_metric).T.round(3)
        df_metric.to_csv(f"{output_dir}/utility-{metric}.csv")
        print(f"  ✓ {metric} results saved")
    
    # Log all utility metrics to wandb
    if wandb.run is not None:
        wandb.log(wandb_metrics)
    
    print(f"\n✓ Utility results saved to {output_dir}/utility-*.csv")


def analyze_privacy_icd(datasets_icd, cfg, output_dir, data_name):
    """Analyze privacy for ICD codes using distance-based metrics."""
    if 'target' not in datasets_icd:
        print(f"  Warning: No ICD code dataset found for {data_name}")
        return None

    print("\n" + "="*50)
    print("PRIVACY ANALYSIS: ICD CODES")
    print("="*50)
    
    # Extract ICD codes
    codes = {}
    for k, dataset in datasets_icd.items():
        codes[k] = [v for p in dataset for v in p["visits"] if len(v) > 0]
        random.seed(42)
        random.shuffle(codes[k])
        print(f"  {k}: {len(codes[k])} visit sequences")
    
    # Compute distance matrices
    print(f"\nComputing distance matrices (N={cfg.n_samples_privacy})...")
    cache_train = f"temp/D_{data_name}_train.npy"
    cache_test = f"temp/D_{data_name}_test.npy"
    
    if os.path.exists(cache_train):
        D_syn_train = np.load(cache_train)
        D_syn_test = np.load(cache_test)
        print("  Loaded from cache")
    else:
        D_syn_train = pairwise_hamming_distance(
            codes['target'][:cfg.n_samples_privacy], 
            codes['train'][:cfg.n_samples_privacy]
        )
        D_syn_test = pairwise_hamming_distance(
            codes['target'][:cfg.n_samples_privacy], 
            codes['test'][:cfg.n_samples_privacy]
        )
        
        os.makedirs("temp", exist_ok=True)
        np.save(cache_train, D_syn_train)
        np.save(cache_test, D_syn_test)
        print("  Computed and cached")
    
    # Compute privacy metrics
    min_d_syn_train = D_syn_train.min(axis=1)
    min_d_syn_test = D_syn_test.min(axis=1)
    
    kde_train = gaussian_kde(min_d_syn_train)
    kde_test = gaussian_kde(min_d_syn_test)
    
    max_val = max(max(min_d_syn_train), max(min_d_syn_test))
    x = np.linspace(0, max_val, 1000)
    y_train = kde_train(x)
    y_test = kde_test(x)
    
    jsd = jensenshannon(y_train, y_test)
    wd = wasserstein_distance(y_train, y_test)
    auroc = roc_auc_score(
        np.concatenate([np.ones_like(min_d_syn_train), np.zeros_like(min_d_syn_test)]),
        np.concatenate([
            [kde_train(x) for x in min_d_syn_train],
            [kde_test(x) for x in min_d_syn_test]
        ])
    )
    
    print(f"\n  JSD: {jsd:.4f}, WD: {wd:.4f}, AUROC: {auroc:.4f}")
    
    # Create privacy distance histogram
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=min_d_syn_train, histnorm='probability', 
        name=f'{data_name}-train', opacity=0.75
    ))
    fig.add_trace(go.Histogram(
        x=min_d_syn_test, histnorm='probability', 
        name=f'{data_name}-test', opacity=0.75
    ))
    
    fig.update_layout(
        barmode='overlay',
        title=f"Privacy Distance Distribution - {data_name}",
        xaxis_title="Minimum Distance to Real Data",
        yaxis_title="Probability",
        template="plotly"
    )
    
    # Save plot locally
    plot_path = f"{output_dir}/privacy-icd-distances.html"
    fig.write_html(plot_path)
    print(f"  ✓ Saved privacy plot to {plot_path}")
    
    # Log to wandb
    if wandb.run is not None:
        wandb.log({
            'privacy-icd/JSD': jsd,
            'privacy-icd/WD': wd,
            'privacy-icd/AUROC': auroc,
            'privacy-icd/distance-plot': wandb.Html(fig.to_html())
        })
    
    # Save results
    df_privacy_icd = pd.DataFrame({
        'JSD': [jsd], 
        'WD': [wd], 
        'AUROC': [auroc]
    }, index=[data_name]).round(4)
    
    df_privacy_icd.to_csv(f"{output_dir}/priv-icd.csv")
    print(f"\n✓ ICD privacy results saved to {output_dir}/priv-icd.csv")
    
    return df_privacy_icd


def analyze_privacy_ts(X, y, cfg, output_dir, data_name):
    """Analyze privacy for time series using MIA and NNAA metrics."""
    print("\n" + "="*50)
    print("PRIVACY ANALYSIS: TIME SERIES")
    print("="*50)
    
    CONT_VARS = list(cfg.continuous_vars)
    LL = 4 * len(CONT_VARS)
    
    all_metrics_mia = []
    all_metrics_nnaa = []
    
    for random_state in tqdm(cfg.random_seeds, desc="Random states"):
        Xy = {}
        
        for k in ['train', 'test', 'target']:
            Xy[k] = pd.concat([X[k].fillna(0).iloc[:, :LL], y[k]], axis=1)
            Xy[k] = Xy[k].sample(cfg.n_samples_privacy, random_state=random_state, replace=False)
            Xy[k] = Xy[k] + np.random.normal(0, 0.00001, Xy[k].shape)
        
        REAL = Xy['train'].values
        FAKE = Xy['target'].values
        TEST = Xy['test'].values
        
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
                wandb_metrics[f'privacy-ts/MIA/{col}'] = float(df_mia[col].iloc[0])
            except:
                pass
        
        # Log NNAA metrics
        for col in df_nnaa.columns:
            try:
                wandb_metrics[f'privacy-ts/NNAA/{col}'] = float(df_nnaa[col].iloc[0])
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
            data_tsne[k] = X[k].fillna(0).values
    
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


def load_promptehr_data(path, metadata):
    """Load PromptEHR data in special dill format and convert to standard format."""
    print("Loading PromptEHR data (special format)...")
    
    codeToIndex = metadata['codeToId']
    n_phe_labels = 25
    
    temp = dill.load(open(path, 'rb'))
    
    data = []
    for v in tqdm(temp['visit'], desc="Processing PromptEHR"):
        code_diags = [codeToIndex[temp['voc']['diag_voc'].idx2word[x]] for x in v[0][0]]
        code_procs = [codeToIndex[temp['voc']['pro_voc'].idx2word[x]] for x in v[0][1]]
        
        data.append({
            'visits': [code_diags + code_procs],
            'labels': np.zeros(n_phe_labels),
            'codes': [code_diags + code_procs],  # For time series compatibility
            'labels_phe': [np.zeros(n_phe_labels)],
            'labels_ihm': [0],
            'covars': [[]],
            'ts': [[]]
        })
    
    print(f"✓ Loaded PromptEHR with {len(data)} patients")
    return data


class Voc(object):
    """Vocabulary class for PromptEHR data format."""
    def __init__(self):
        self.idx2word = {}
        self.word2idx = {}

    def add_sentence(self, sentence):
        for word in sentence:
            if word not in self.word2idx:
                self.idx2word[len(self.word2idx)] = word
                self.word2idx[word] = len(self.word2idx)


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
    
    # Check if this is a baseline model
    is_promptehr = 'pehr' in data_name.lower()
    is_rtsgan = 'rtsgan' in data_name.lower()
    
    # Setup output directory
    output_dir = f'RESULTS/{data_name}'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}\n")
    
    # Initialize wandb
    if cfg.get('wandb', {}).get('project'):
        wandb.init(
            project=cfg.wandb.project,
            name=f"results-{data_name}",
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=['evaluation', 'results', data_name]
        )
        print(f"✓ W&B initialized: {cfg.wandb.project}\n")
    else:
        print("⚠ W&B not configured, skipping logging\n")
    
    # ==================== LOAD METADATA ====================
    print("Loading metadata...")
    metadata = pickle.load(open(cfg.paths.meta, 'rb'))
    
    indexToCode = metadata['idToCode']
    codeToIndex = metadata['codeToId']
    var2id = metadata['var2id']
    id2var = {v: k for k, v in var2id.items()}
    
    print("✓ Metadata loaded\n")
    
    # ==================== LOAD ICD DATASETS ====================
    print("Loading ICD code datasets...")
    
    # Load training and test sets
    datasets_icd = {
        'train': pickle.load(open(cfg.paths.real.format(SPLIT='train'), 'rb')),
        'test': pickle.load(open(cfg.paths.real.format(SPLIT='test'), 'rb')),
    }
    
    # Load the target dataset
    if is_real_split:
        # If evaluating a real split (train/test/val), it's already loaded or we load it
        if data_name not in datasets_icd:
            datasets_icd[data_name] = pickle.load(open(cfg.paths.real.format(SPLIT=data_name), 'rb'))
        # Use the same data as target
        datasets_icd['target'] = datasets_icd[data_name]
    elif is_promptehr:
        # Load PromptEHR data (special format)
        datasets_icd['target'] = load_promptehr_data(cfg.paths.syn.format(RUN_NAME=data_name), metadata)
    else:
        # Load synthetic dataset
        try:
            datasets_icd['target'] = pickle.load(open(cfg.paths.syn.format(RUN_NAME=data_name), 'rb'))
        except Exception as e:
            print(f"  Warning: Could not load synthetic ICD code dataset for {data_name}")
    
    # Extract ICD codes only
    for k in datasets_icd.keys():
        if k == 'target' and is_promptehr:
            # PromptEHR already has correct format
            continue
        datasets_icd[k] = [
            {'visits': p['codes'], 'labels': p['labels_phe']} 
            for p in datasets_icd[k]
        ]
    
    # Remove empty patients
    for k in datasets_icd.keys():
        len_before = len(datasets_icd[k])
        datasets_icd[k] = [x for x in datasets_icd[k] if len(x['visits']) > 0]
        print(f"  {k}: {len_before} → {len(datasets_icd[k])}")
    
    print("✓ ICD datasets loaded\n")
    
    # ==================== LOAD TIME SERIES DATASETS ====================
    print("Loading time series datasets...")
    
    # Load training and test sets
    datasets = {
        'train': pickle.load(open(cfg.paths.real.format(SPLIT='train'), 'rb')),
        'test': pickle.load(open(cfg.paths.real.format(SPLIT='test'), 'rb')),
    }
    
    # Load the target dataset
    if is_real_split:
        if data_name not in datasets:
            datasets[data_name] = pickle.load(open(cfg.paths.real.format(SPLIT=data_name), 'rb'))
        datasets['target'] = datasets[data_name]
    elif is_promptehr:
        # PromptEHR doesn't have time series data - use the ICD version
        datasets['target'] = datasets_icd['target']
    else:
        # Load synthetic dataset
        try:
            datasets['target'] = pickle.load(open(cfg.paths.syn.format(RUN_NAME=data_name), 'rb'))
        except Exception as e:
            print(f"  Warning: Could not load time series data for {data_name}: {e}")
            datasets['target'] = None
    
    # Remove empty patients
    for k in datasets.keys():
        if datasets[k] is None:
            print(f"  {k}: Dataset not available")
            continue
        len_before = len(datasets[k])
        datasets[k] = [x for x in datasets[k] if len(x['codes']) > 0]
        print(f"  {k}: {len_before} → {len(datasets[k])}")
    
    print("✓ Time series datasets loaded\n")
    
    # ==================== CONVERT TO DATAFRAMES ====================
    print("Converting to DataFrames...")
    
    datasets_df_ts = {}
    datasets_df_static = {}
    
    # Create label columns
    COL_LABELS = [f'label_phe_{i}' for i in range(cfg.n_phe_labels)] + ['label_ihm']
    
    for k in ['train', 'test']:
        print(f"  Processing {k}...")
        datasets_df_ts[k] = pd.read_csv(cfg.paths.ts.format(SPLIT=k)).rename(
            columns={"RecordID": 'id', 'Time': 'Hours'}
        )
        
        labels = ['label_ihm'] + [f'label_phe_{i}' for i in range(cfg.n_phe_labels)]
        datasets_df_static[k] = datasets_df_ts[k][['id', 'Age', 'Gender'] + labels]\
            .groupby('id').first().reset_index()
        datasets_df_ts[k] = datasets_df_ts[k].drop(columns=['Age', 'Gender'] + labels)
    
    # Process target dataset
    if datasets.get('target') is None:
        print(f"  Warning: No time series data available for {data_name}")
        print(f"  Skipping time series analyses")
        skip_timeseries = True
    else:
        skip_timeseries = False
        
        if is_real_split:
            # If it's a real split and not already processed
            if data_name not in datasets_df_ts:
                print(f"  Processing {data_name}...")
                datasets_df_ts[data_name] = pd.read_csv(cfg.paths.ts.format(SPLIT=data_name)).rename(
                    columns={"RecordID": 'id', 'Time': 'Hours'}
                )
                
                labels = ['label_ihm'] + [f'label_phe_{i}' for i in range(cfg.n_phe_labels)]
                datasets_df_static[data_name] = datasets_df_ts[data_name][['id', 'Age', 'Gender'] + labels]\
                    .groupby('id').first().reset_index()
                datasets_df_ts[data_name] = datasets_df_ts[data_name].drop(columns=['Age', 'Gender'] + labels)
            
            # Reference as target
            datasets_df_ts['target'] = datasets_df_ts[data_name]
            datasets_df_static['target'] = datasets_df_static[data_name]
            
        elif is_rtsgan:
            # Load pre-processed files for RTSGAN
            print(f"  Processing {data_name} (RTSGAN format)...")
            try:
                datasets_df_ts['target'] = pd.read_csv(f"{cfg.paths.ts_cache}/df-ts-{data_name}.csv").rename(
                    columns={"RecordID": 'id', 'Time': 'Hours'}
                )
                datasets_df_static['target'] = pd.read_csv(f"{cfg.paths.ts_cache}/df-static-{data_name}.csv").rename(
                    columns={"RecordID": 'id', 'Label': 'label_ihm'}
                ).drop(columns=['seq_len'], errors='ignore')
                
                # Fix label naming
                if 'phe_0' in datasets_df_static['target'].columns:
                    print(f"  Fixing labels for {data_name}")
                    datasets_df_static['target'].rename(
                        columns={f'phe_{i}': f'label_phe_{i}' for i in range(cfg.n_phe_labels)}, 
                        inplace=True
                    )
            except Exception as e:
                print(f"  Error loading RTSGAN data: {e}")
                skip_timeseries = True
                
        else:
            # Process other synthetic data (timehr, SynEHRgy, HALO, etc.)
            print(f"  Processing {data_name}...")
            dfs_ts, dfs_covar = get_df_ts_covars(data_name, datasets['target'], metadata, cfg.paths.ts_cache)
            datasets_df_static['target'] = dfs_covar
            datasets_df_ts['target'] = dfs_ts
            
            # Fix label naming inconsistencies
            if 'phe_0' in datasets_df_static['target'].columns:
                print(f"  Fixing labels for {data_name}")
                datasets_df_static['target'].rename(
                    columns={f'phe_{i}': f'label_phe_{i}' for i in range(cfg.n_phe_labels)}, 
                    inplace=True
                )
            
            if 'Label' in datasets_df_static['target'].columns:
                print(f"  Fixing label_ihm for {data_name}")
                datasets_df_static['target'].rename(columns={'Label': 'label_ihm'}, inplace=True)
    
    # Filter to time window
    if not skip_timeseries:
        print(f"\nFiltering to first {cfg.time_window_hours} hours...")
        for k in datasets_df_ts.keys():
            before_count = len(datasets_df_ts[k])
            datasets_df_ts[k] = datasets_df_ts[k][datasets_df_ts[k]['Hours'] < cfg.time_window_hours]
            after_count = len(datasets_df_ts[k])
            print(f"  {k}: {before_count} → {after_count} measurements")
    
    print("✓ DataFrame conversion complete\n")
    
    # ==================== CREATE TIME SERIES EMBEDDINGS ====================
    if not skip_timeseries:
        print("Creating time series embeddings...")
        
        X, y = {}, {}
        CONT_VARS = list(cfg.continuous_vars)
        
        for k, df_ts in datasets_df_ts.items():
            print(f"  Processing {k}...")
            df_static = datasets_df_static[k]
            X[k], y[k] = genTSembeddings(df_ts, df_static, CONT_VARS, COL_LABELS)
        print("✓ Time series embeddings created\n")
    else:
        X, y = None, None
        print("✓ Skipping time series embeddings (no data)\n")
    
    # ==================== RUN ANALYSES ====================
    
    # Fidelity - ICD
    analyze_fidelity_icd(datasets_icd, metadata, output_dir, cfg, data_name)
    
    # Fidelity - Time Series (skip if no time series data)
    if not skip_timeseries:
        analyze_fidelity_ts(datasets_df_ts, datasets_df_static, X, y, cfg, output_dir, data_name)
    else:
        print("\n⚠ Skipping time series fidelity analysis (no data)\n")
    
    # Utility (skip if no time series data)
    if not skip_timeseries:
        analyze_utility(X, y, cfg, output_dir, data_name)
    else:
        print("\n⚠ Skipping utility analysis (no time series data)\n")
    
    # Privacy - ICD
    analyze_privacy_icd(datasets_icd, cfg, output_dir, data_name)
    
    # Privacy - Time Series (skip if no time series data)
    if not skip_timeseries:
        analyze_privacy_ts(X, y, cfg, output_dir, data_name)
    else:
        print("\n⚠ Skipping time series privacy analysis (no data)\n")
    
    # t-SNE Visualization (skip if no time series data)
    if not skip_timeseries:
        create_tsne_visualization(X, cfg, output_dir, data_name)
    else:
        print("\n⚠ Skipping t-SNE visualization (no data)\n")
    
    # ==================== DONE ====================
    print("\n" + "="*70)
    print("RESULTS GENERATION COMPLETE")
    print("="*70)
    print(f"\nAll results saved to: {output_dir}/")
    print("\nGenerated files:")
    print("\n  CSV Results:")
    print("  - fid-icd-ngram.csv: ICD n-gram fidelity")
    if not skip_timeseries:
        print("  - fid-ts.csv: Time series fidelity (PRDC + TCD)")
        print("  - utility-*.csv: Utility metrics for downstream tasks")
    print("  - priv-icd.csv: ICD privacy metrics")
    if not skip_timeseries:
        print("  - privacy-ts-mia.csv: Time series MIA privacy")
        print("  - privacy-ts-nnaa.csv: Time series NNAA privacy")
    
    print("\n  Plots:")
    print("  - 1-gram.html, 2-gram.html, 3-gram.html: N-gram scatter plots")
    if not skip_timeseries:
        print(f"  - corr_{data_name}.png: Correlation confusion matrix")
        print(f"  - occ-{data_name}.html: Co-occurrence matrix")
        print("  - utility-*.html: Utility curves for each metric")
        print("  - tsne.html: t-SNE visualization")
    print("  - privacy-icd-distances.html: Privacy distance distribution")
    
    if skip_timeseries:
        print("\n⚠ Note: Time series analyses were skipped (no time series data available)")
    
    # Finish wandb run
    if wandb.run is not None:
        wandb.finish()
        print("\n✓ W&B run finished")
        print(f"  View results at: {wandb.run.url}")


if __name__ == "__main__":
    main()
