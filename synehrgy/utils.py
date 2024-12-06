import os
from typing import Any, Dict, List, Tuple

import torch
import numpy as np
import random
import pickle
from tqdm import tqdm
import json
# import matplotlib.pyplot as plt
import torchvision

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import itertools
import wandb


# def modify_dataset(dataset_orig):

#     n_before = len(dataset_orig)
#     dataset_orig = [
#         patient
#         for patient in dataset_orig
#         if sum([len(visit) for visit in patient["visits"]])
#         + len(patient["visits"])
#         + config.label_vocab_size
#         + 3
#         < config.n_ctx
#     ]
#     n_after = len(dataset_orig)
#     print(f"Removed {n_before - n_after} patients from dataset")

#     dataset = []
#     for orig_ehr in dataset_orig:
#         new_ehr = [config.total_vocab_size - 1] * config.n_ctx  # Pad Codes
#         new_ehr[0] = config.code_vocab_size + config.label_vocab_size  # Start Record
#         idx = 1

#         # Add Labels
#         for l in orig_ehr["labels"].nonzero()[0]:
#             new_ehr[idx] = l + config.code_vocab_size
#             idx += 1

#         new_ehr[idx] = (
#             config.code_vocab_size + config.label_vocab_size + 1
#         )  # End Labels
#         idx += 1

#         # Add Visits
#         for v in orig_ehr["visits"]:
#             for c in v:

#                 new_ehr[idx] = c
#                 idx += 1
#             new_ehr[idx] = (
#                 config.code_vocab_size + config.label_vocab_size + 2
#             )  # End Visit
#             idx += 1

#         new_ehr[idx] = (
#             config.code_vocab_size + config.label_vocab_size + 3
#         )  # End Record
#         dataset.append(new_ehr)
#     return dataset


def get_batch(loc, batch_size, dataset):

    ehr = dataset[loc : loc + batch_size]

    batch_ehr = np.array(ehr)
    return batch_ehr


def dataset_ngram(dataset, n=1, th=[0, 1], batch_size=512, pad_token_id=0):

    from collections import Counter
    from typing import List, Tuple, Dict

    # Function to compute n-grams
    def compute_ngrams(tokens: List[int], n: int) -> List[Tuple[int]]:
        return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

    # Function to update n-gram counts for a batch
    def update_ngram_counts(
        batch: torch.Tensor, pad_token_id: int, n: int, global_ngram_counts: Counter
    ) -> None:
        for i in range(batch.size(0)):
            tokens = [token for token in batch[i].tolist() if token != pad_token_id]
            ngrams = compute_ngrams(tokens, n)
            global_ngram_counts.update(ngrams)

    # Function to compute combined n-gram probabilities
    def compute_combined_ngram_probabilities(
        global_ngram_counts: Counter,
    ) -> Dict[Tuple[int], float]:
        total_ngrams = sum(global_ngram_counts.values())
        combined_ngram_prob = {
            ngram: freq / total_ngrams for ngram, freq in global_ngram_counts.items()
        }
        return combined_ngram_prob

    global_ngram_counts = Counter()
    for i in tqdm(range(0, len(dataset), batch_size), desc="Eval"):

        batch_ehr = get_batch(i, batch_size, dataset)
        batch_ehr = torch.tensor(batch_ehr, dtype=torch.long).to("cuda")

        # Update global n-gram counts
        update_ngram_counts(batch_ehr, pad_token_id, n, global_ngram_counts)

    # Compute combined n-gram probabilities after processing all batches
    combined_probs = compute_combined_ngram_probabilities(global_ngram_counts)

    # Filter n-grams based on threshold
    combined_probs = {
        ngram: prob
        for ngram, prob in combined_probs.items()
        if prob >= th[0] and prob <= th[1]
    }

    return combined_probs


def plot_scatter(probs_x, probs_y, log_scale=False, title="unigram"):
    # Find the intersection of unigrams
    common_unigrams = set(probs_x.keys()).intersection(set(probs_y.keys()))
    print(f"Number of data1 {title}: {len(probs_x)}")
    print(f"Number of data2 {title}: {len(probs_y)}")
    print(f"Number of common {title}: {len(common_unigrams)}")

    # Extract corresponding probabilities
    x_probs = [probs_x[unigram] for unigram in common_unigrams]
    y_probs = [probs_y[unigram] for unigram in common_unigrams]

    # Create scatterplot
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x_probs,
            y=y_probs,
            mode="markers+text",
            # text=[str(unigram) for unigram in unigrams],
            textposition="top center",
            marker=dict(
                size=5,
                color="rgba(152, 0, 0, .8)",
                line=dict(
                    width=0,
                ),
            ),
            name=title,
        )
    )

    # draw line y=x
    max_val = max(max(x_probs), max(x_probs))
    _ = fig.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode="lines",
            name="y=x",
            line=dict(color="royalblue", width=3, dash="dash"),
        )
    )

    fig.update_layout(
        title=f"Scatterplot of {title} Probabilities",
        xaxis_title="Train Probabilities",
        yaxis_title="Validation Probabilities",
        template="plotly_white",
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(scaleanchor="x", scaleratio=1),
        height=400,
        width=400,
        showlegend=False,
    )

    # set xlim and ylim
    # fig.update_xaxes(range=[-0.0001, max_val])
    # fig.update_yaxes(range=[-0.0001, max_val])

    if log_scale:
        # # logarithmic axis
        _ = fig.update_xaxes(type="log")
        _ = fig.update_yaxes(type="log")
    else:
        # set max_val to 95th percentile
        max_val = np.percentile((x_probs + y_probs), 99) * 0 + 0.01

        fig.update_xaxes(range=[-0.0001, max_val])
        fig.update_yaxes(range=[-0.0001, max_val])

    return fig


import numpy as np
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import entropy, pearsonr
from sklearn.metrics import r2_score


def compute_metrics(probs_train, probs_val):
    # Find the intersection of unigrams
    common_unigrams = set(probs_train.keys()).intersection(set(probs_val.keys()))

    if len(common_unigrams) < 10:
        return {
            "KL Divergence": 0,
            "JS Divergence": 0,
            "Cosine Similarity": 0,
            "Euclidean Distance": 0,
            "Pearson Correlation": 0,
            "R2 Score": 0,
            "common_unigrams": 0,
        }

    # Extract corresponding probabilities
    train_probs = np.array([probs_train[unigram] for unigram in common_unigrams])
    val_probs = np.array([probs_val[unigram] for unigram in common_unigrams])

    # Normalize probabilities
    train_probs /= train_probs.sum()
    val_probs /= val_probs.sum()

    # Compute KL Divergence
    kl_div = entropy(train_probs, val_probs)

    # Compute JS Divergence
    m = 0.5 * (train_probs + val_probs)
    js_div = 0.5 * (entropy(train_probs, m) + entropy(val_probs, m))

    # Compute Cosine Similarity
    cosine_sim = 1 - cosine(train_probs, val_probs)

    # Compute Euclidean Distance
    euclidean_dist = euclidean(train_probs, val_probs)

    # Compute Pearson Correlation
    correlation, _ = pearsonr(train_probs, val_probs)

    # Compute R2 Score
    r2 = r2_score(train_probs, val_probs)

    # compute R2 score in the log space
    r2_log = r2_score(np.log(train_probs), np.log(val_probs))
    return {
        "KL Divergence": kl_div,
        "JS Divergence": js_div,
        "Cosine Similarity": cosine_sim,
        "Euclidean Distance": euclidean_dist,
        "Pearson Correlation": correlation,
        "R2 Score": r2,
        "R2 Score Log": r2_log,
        "common_unigrams": len(common_unigrams) / len(probs_train) * 100,
    }


def generate_statistics(ehr_dataset):
    N_LABELS = len(ehr_dataset[0]["labels"])
    label_mapping = {i: f"L_{i}" for i in range(N_LABELS)}
    label_mapping[N_LABELS] = "Overall"

    stats = {}
    label_counts = {}
    for i in tqdm(range(N_LABELS + 1)):
        label_stats = {}
        d = (
            [p for p in ehr_dataset if p["labels"][i] == 1]
            if i < N_LABELS
            else ehr_dataset
        )
        label_counts[label_mapping[i]] = len(d)

    for i in tqdm(range(N_LABELS, N_LABELS + 1)):
        label_stats = {}
        d = (
            [p for p in ehr_dataset if p["labels"][i] == 1]
            if i < N_LABELS
            else ehr_dataset
        )
        label_counts[label_mapping[i]] = len(d)

        aggregate_stats = {}
        record_lens = [len(p["visits"]) for p in d]
        visit_lens = [len(v) for p in d for v in p["visits"]]
        avg_record_len = np.mean(record_lens)
        std_record_len = np.std(record_lens)
        avg_visit_len = np.mean(visit_lens)
        std_visit_len = np.std(visit_lens)
        aggregate_stats["Record Length Mean"] = avg_record_len
        aggregate_stats["Record Length Standard Deviation"] = std_record_len
        aggregate_stats["Visit Length Mean"] = avg_visit_len
        aggregate_stats["Visit Length Standard Deviation"] = std_visit_len
        label_stats["Aggregate"] = aggregate_stats

        code_stats = {}
        n_records = len(record_lens)
        n_visits = len(visit_lens)
        record_code_counts = {}  # how many patients have a code
        visit_code_counts = {}  # how many visits have a code
        record_bigram_counts = (
            {}
        )  # how many patients have (c1,c2) in one of their visits
        visit_bigram_counts = {}  # how many times we have (c1,c2) in a visit
        record_sequential_bigram_counts = (
            {}
        )  # how many patients have c1 (visit i) and c2 (visit i+1)
        visit_sequential_bigram_counts = (
            {}
        )  # how many times we have c1 (visit i) and c2 (visit i+1)

        visit_3gram_counts = {}
        for p in tqdm(d, desc="Generating Statistics"):
            patient_codes = set()
            patient_bigrams = set()
            sequential_bigrams = set()
            for j in range(len(p["visits"])):
                v = p["visits"][j]  # codes in an admission
                for c in v:
                    visit_code_counts[c] = (
                        1 if c not in visit_code_counts else visit_code_counts[c] + 1
                    )
                    patient_codes.add(c)
                for cs in itertools.combinations(v, 2):
                    cs = list(cs)
                    # cs.sort()  # why sort?!!!
                    cs = tuple(cs)
                    visit_bigram_counts[cs] = (
                        1
                        if cs not in visit_bigram_counts
                        else visit_bigram_counts[cs] + 1
                    )
                    patient_bigrams.add(cs)
                # compute 3gram_counts
                for cs in itertools.combinations(v, 3):
                    cs = list(cs)
                    # cs.sort()
                    cs = tuple(cs)
                    visit_3gram_counts[cs] = (
                        1
                        if cs not in visit_3gram_counts
                        else visit_3gram_counts[cs] + 1
                    )
                if j > 0:  # for multiple admissions obly
                    v0 = p["visits"][j - 1]
                    for c0 in v0:
                        for c in v:
                            sc = (c0, c)
                            visit_sequential_bigram_counts[sc] = (
                                1
                                if sc not in visit_sequential_bigram_counts
                                else visit_sequential_bigram_counts[sc] + 1
                            )
                            sequential_bigrams.add(sc)
            for c in patient_codes:
                record_code_counts[c] = (
                    1 if c not in record_code_counts else record_code_counts[c] + 1
                )
            for cs in patient_bigrams:
                record_bigram_counts[cs] = (
                    1
                    if cs not in record_bigram_counts
                    else record_bigram_counts[cs] + 1
                )
            for sc in sequential_bigrams:
                record_sequential_bigram_counts[sc] = (
                    1
                    if sc not in record_sequential_bigram_counts
                    else record_sequential_bigram_counts[sc] + 1
                )

        # remove counts that are less than 5

        def remove_small_counts(counts):
            return {k: v for k, v in counts.items() if v > 5}

        # record_code_counts = remove_small_counts(record_code_counts)
        # visit_code_counts = remove_small_counts(visit_code_counts)
        # record_bigram_counts = remove_small_counts(record_bigram_counts)
        # visit_bigram_counts = remove_small_counts(visit_bigram_counts)
        # visit_3gram_counts = remove_small_counts(visit_3gram_counts)
        # record_sequential_bigram_counts = remove_small_counts(
        #     record_sequential_bigram_counts
        # )
        # visit_sequential_bigram_counts = remove_small_counts(
        #     visit_sequential_bigram_counts
        # )

        record_code_probs = {
            c: record_code_counts[c] / n_records for c in record_code_counts
        }
        visit_code_probs = {
            c: visit_code_counts[c] / n_visits for c in visit_code_counts
        }
        record_bigram_probs = {
            cs: record_bigram_counts[cs] / n_records for cs in record_bigram_counts
        }
        visit_bigram_probs = {
            cs: visit_bigram_counts[cs] / n_visits for cs in visit_bigram_counts
        }
        visit_3gram_probs = {
            cs: visit_3gram_counts[cs] / n_visits for cs in visit_3gram_counts
        }
        record_sequential_bigram_probs = {
            sc: record_sequential_bigram_counts[sc] / n_records
            for sc in record_sequential_bigram_counts
        }
        visit_sequential_bigram_probs = {
            sc: visit_sequential_bigram_counts[sc] / (n_visits - len(d))
            for sc in visit_sequential_bigram_counts
        }
        code_stats["Per Record Code Probabilities"] = record_code_probs
        code_stats["Per Visit Code Probabilities"] = visit_code_probs
        code_stats["Per Record Bigram Probabilities"] = record_bigram_probs
        code_stats["Per Visit Bigram Probabilities"] = visit_bigram_probs
        code_stats["Per Visit 3gram Probabilities"] = visit_3gram_probs
        code_stats["Per Record Sequential Visit Bigram Probabilities"] = (
            record_sequential_bigram_probs
        )
        code_stats["Per Visit Sequential Visit Bigram Probabilities"] = (
            visit_sequential_bigram_probs
        )
        label_stats["Probabilities"] = code_stats
        stats[label_mapping[i]] = label_stats
    label_probs = {l: label_counts[l] / n_records for l in label_counts}
    stats["Label Probabilities"] = label_probs
    return stats



def generate_statistics2(ehr_dataset):

    N_LABELS = len(ehr_dataset[0]["labels"])
    label_mapping = {i: f"L_{i}" for i in range(N_LABELS)}
    label_mapping[N_LABELS] = "Overall"

    stats = {}
    label_counts = {}
    for i in tqdm(range(N_LABELS + 1)):
        label_stats = {}
        d = (
            [p for p in ehr_dataset if p["labels"][i] == 1]
            if i < N_LABELS
            else ehr_dataset
        )
        label_counts[label_mapping[i]] = len(d)

    for i in tqdm(range(N_LABELS, N_LABELS + 1)):
        label_stats = {}
        d = (
            [p for p in ehr_dataset if p["labels"][i] == 1]
            if i < N_LABELS
            else ehr_dataset
        )
        label_counts[label_mapping[i]] = len(d)




        aggregate_stats = {}
        record_lens = [len(p["visits"]) for p in d]
        visit_lens = [len(v[0]) for p in d for v in p["visits"] if v[1] == []]
        visit_gaps = [v[3][0] for p in d for v in p["visits"] if v[1] != []]
        avg_record_len = np.mean(record_lens)
        std_record_len = np.std(record_lens)
        avg_visit_len = np.mean(visit_lens)
        std_visit_len = np.std(visit_lens)
        avg_visit_gap = np.mean(visit_gaps)
        std_visit_gap = np.std(visit_gaps)
        aggregate_stats["Record Length Mean"] = avg_record_len
        aggregate_stats["Record Length Standard Deviation"] = std_record_len
        aggregate_stats["Visit Length Mean"] = avg_visit_len
        aggregate_stats["Visit Length Standard Deviation"] = std_visit_len
        aggregate_stats["Visit Gap Mean"] = avg_visit_gap
        aggregate_stats["Visit Gap Standard Deviation"] = std_visit_gap
        label_stats["Aggregate"] = aggregate_stats

        code_stats = {}
        n_records = len(record_lens)
        n_visits = len(visit_lens)
        record_code_counts = {}
        visit_code_counts = {}
        record_bigram_counts = {}
        visit_bigram_counts = {}
        for p in tqdm(d, desc="Generating Code Stats"):
            patient_codes = set()
            patient_bigrams = set()
            for j in range(len(p["visits"])):
                v = p["visits"][j]
                if v[1] != []:
                    continue

                for c in v[0]:
                    visit_code_counts[c] = (
                        1 if c not in visit_code_counts else visit_code_counts[c] + 1
                    )
                    patient_codes.add(c)
                for cs in itertools.combinations(v[0], 2):
                    cs = list(cs)
                    cs.sort()
                    cs = tuple(cs)
                    visit_bigram_counts[cs] = (
                        1
                        if cs not in visit_bigram_counts
                        else visit_bigram_counts[cs] + 1
                    )
                    patient_bigrams.add(cs)
            for c in patient_codes:
                record_code_counts[c] = (
                    1 if c not in record_code_counts else record_code_counts[c] + 1
                )
            for cs in patient_bigrams:
                record_bigram_counts[cs] = (
                    1
                    if cs not in record_bigram_counts
                    else record_bigram_counts[cs] + 1
                )
        record_code_probs = {
            c: record_code_counts[c] / n_records for c in record_code_counts
        }
        visit_code_probs = {
            c: visit_code_counts[c] / n_visits for c in visit_code_counts
        }
        record_bigram_probs = {
            cs: record_bigram_counts[cs] / n_records for cs in record_bigram_counts
        }
        visit_bigram_probs = {
            cs: visit_bigram_counts[cs] / n_visits for cs in visit_bigram_counts
        }
        code_stats["Per Record Code Probabilities"] = record_code_probs
        code_stats["Per Visit Code Probabilities"] = visit_code_probs
        code_stats["Per Record Bigram Probabilities"] = record_bigram_probs
        code_stats["Per Visit Bigram Probabilities"] = visit_bigram_probs
        
        label_stats["Probabilities"] = code_stats
        stats[label_mapping[i]] = label_stats
    label_probs = {l: label_counts[l] / n_records for l in label_counts}
    stats["Label Probabilities"] = label_probs
    return stats

def generate_plots(
    stats1,
    stats2,
    label1,
    label2,
    types=[
        "Per Visit Code Probabilities",
        "Per Visit Bigram Probabilities",
        "Per Visit 3gram Probabilities",
        "Per Record Code Probabilities",
        "Per Record Bigram Probabilities",
        "Per Visit Sequential Visit Bigram Probabilities",
        "Per Record Sequential Visit Bigram Probabilities",
    ],
):
    N_LABELS = len(stats1['Label Probabilities'])-1
    label_mapping = {i: f"L_{i}" for i in range(N_LABELS)}
    label_mapping[N_LABELS] = "Overall"

    for i in tqdm(range(N_LABELS, N_LABELS + 1)):
        print("\n")
        label = label_mapping[i]
        data1 = stats1[label]["Probabilities"]
        data2 = stats2[label]["Probabilities"]
        for t in types:
            if t not in data1 or t not in data2:
                continue
            probs1 = data1[t]
            probs2 = data2[t]
            keys = set(probs1.keys()).union(set(probs2.keys()))
            values1 = [probs1[k] if k in probs1 else 0 for k in keys]
            values2 = [probs2[k] if k in probs2 else 0 for k in keys]

            # keys2 = set(probs1.keys()).intersection(set(probs2.keys()))
            # values1 = [probs1[k] if k in probs1 else 0 for k in keys2]
            # values2 = [probs2[k] if k in probs2 else 0 for k in keys2]

            r2score = r2_score(values1, values2)
            print(f"{t}: {r2score}")

            # plt.clf()
            # plt.scatter(values1, values2, marker=".", alpha=0.66)
            # maxVal = min(1.1 * max(max(values1), max(values2)), 1.0)
            # plt.xlim([0, maxVal])
            # plt.ylim([0, maxVal])
            # plt.title(f"{label} {t}")
            # plt.xlabel(label1)
            # plt.ylabel(label2)
            # plt.savefig(
            #     f"results/dataset_stats/plots/{label2}_{label.split(':')[-1]}_{t}_adjMax".replace(
            #         " ", "_"
            #     )
            # )

            # use plotly
            df = pd.DataFrame({"x": values1, "y": values2})
            if len(df) > 2000:
                df = df.sample(2000)
            # Calculate maxVal and r2_score
            maxVal = min(1.1 * max(max(values1), max(values2)), 1.0)
            maxVal = min(1.1 * df.max().max(), 1.0)
            r2 = r2_score(values1, values2)

            # Create the scatter plot with plotly express
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(x=df["x"], y=df["y"], mode="markers", name="Data points")
            )
            fig.add_trace(
                go.Scatter(
                    x=[0, maxVal],
                    y=[0, maxVal],
                    mode="lines",
                    line=dict(color="red"),
                    # name="Red line",
                )
            )
            r2_005 = r2_score(df[df["x"] < 0.005]["x"], df[df["x"] < 0.005]["y"])
            # r2 = r2_score(df[df["x"] > 0.005]["x"], df[df["x"] > 0.005]["y"])
            # Update the layout to set the x and y axis limits
            fig.update_layout(
                xaxis=dict(range=[0, maxVal], title=label1),
                yaxis=dict(range=[0, maxVal], title=label2),
            )
            # remove legend
            fig.update_layout(
                showlegend=False,
                title=f"r2: ({r2_005:.3f})({r2:.3f})",
                # autosize=False,
                # width=500,
                # height=500,
                # margin=dict(l=0, r=0, b=0, t=0, pad=0),
            )

            # Save the plot
            wandb.log({f"BiUni/{t}": wandb.Plotly(fig)})



def find_hamming(ehr, dataset, return_all=False, exclude=False):
    min_d = 1e10
    visits = ehr["visits"]
    labels = ehr["labels"]
    all_dist = []
    for p in dataset:

        d = 0 if len(visits) == len(p["visits"]) else 1
        l = p["labels"]
        d += ((labels + l) == 1).sum()
        for i in range(len(visits)):
            v = visits[i]
            if i >= len(p["visits"]):
                d += len(v)
            else:
                v2 = p["visits"][i]
                d += len(v) + len(v2) - (2 * len(set(v).intersection(v2)))

        if exclude and p["subject_id"] == ehr["subject_id"]:
            d = 100
        if return_all:
            # replace 0 with 500 list if exits
            # if 0 in all_dist:
            #     all_dist[all_dist.index(0)] = 500

            all_dist.append(d)
        min_d = d if d < min_d and d > -1 else min_d
    if return_all:
        return min_d, all_dist
    else:
        return min_d


def calc_nnaar(train, evaluation, synthetic):
    NUM_SAMPLES = len(train)
    val1 = 0
    val2 = 0
    val3 = 0
    val4 = 0
    val5 = 0
    val6 = 0
    all_det = []
    all_dte = []
    all_dee = []
    all_dtt = []

    all_des = []
    all_dse = []
    all_dst = []
    all_dts = []
    all_dss = []

    for p in tqdm(evaluation):
        des, all_dist = find_hamming(p, synthetic, return_all=True)
        all_des.append(all_dist)
        # if des != min(all_dist):
        #     term
        dee, all_dist = find_hamming(p, evaluation, return_all=True, exclude=True)
        all_dee.append(all_dist)
        # if dee != min(all_dist):
        #     term
        if des > dee:
            val1 += 1

        det, all_dist = find_hamming(p, train, return_all=True)

        # ADDED
        all_det.append(all_dist)
        if det > dee:
            val5 += 1

    for p in tqdm(train):
        dts, all_dist = find_hamming(p, synthetic, return_all=True)
        all_dts.append(all_dist)
        dtt, all_dist = find_hamming(p, train, return_all=True, exclude=True)
        all_dtt.append(all_dist)
        if dts > dtt:
            val3 += 1
        dte, all_dist = find_hamming(p, evaluation, return_all=True)

        # ADDED
        all_dte.append(all_dist)
        if dte > dtt:
            val6 += 1

    for p in tqdm(synthetic):
        dse, all_dist = find_hamming(p, evaluation, return_all=True)
        all_dse.append(all_dist)
        dst, all_dist = find_hamming(p, train, return_all=True)
        all_dst.append(all_dist)
        dss, all_dist = find_hamming(p, synthetic, return_all=True, exclude=True)
        all_dss.append(all_dist)

        if dse > dss:
            val2 += 1
        if dst > dss:
            val4 += 1

    def compute_temp(all_det, all_dte, all_dee, all_dtt):

        det = np.array(all_det)
        dte = np.array(all_dte)
        dee = np.array(all_dee)
        dtt = np.array(all_dtt)
        # set diagonal to 100
        # np.fill_diagonal(dee, 100)
        # np.fill_diagonal(det, 100)
        # np.fill_diagonal(dte, 100)
        # np.fill_diagonal(dtt, 100)

        m_et = (np.min(dee, axis=1) < np.min(det, axis=1)).sum()
        m_te = (np.min(dtt, axis=1) < np.min(dte, axis=1)).sum()

        return m_et, m_te

    # print("val5", (np.min(dee, axis=1) < np.min(det, axis=1)).sum())
    # print("val6", (np.min(dtt, axis=1) < np.min(dte, axis=1)).sum())

    pickle.dump(all_det, open("all_det.pkl", "wb"))
    pickle.dump(all_dte, open("all_dte.pkl", "wb"))
    pickle.dump(all_dee, open("all_dee.pkl", "wb"))
    pickle.dump(all_dtt, open("all_dtt.pkl", "wb"))

    val1 = val1 / NUM_SAMPLES
    val2 = val2 / NUM_SAMPLES
    val3 = val3 / NUM_SAMPLES
    val4 = val4 / NUM_SAMPLES

    val5 = val5 / NUM_SAMPLES
    val6 = val6 / NUM_SAMPLES

    # e s
    val11, val22 = compute_temp(all_des, all_dse, all_dee, all_dss)
    # s e
    # t s
    val33, val44 = compute_temp(all_dts, all_dst, all_dtt, all_dss)
    # s t
    # e t
    val55, val66 = compute_temp(all_det, all_dte, all_dee, all_dtt)
    # t e

    val11 = val11 / NUM_SAMPLES
    val22 = val22 / NUM_SAMPLES
    val33 = val33 / NUM_SAMPLES
    val44 = val44 / NUM_SAMPLES
    val55 = val55 / NUM_SAMPLES
    val66 = val66 / NUM_SAMPLES

    print("val1", val1)
    print("val2", val2)
    print("val3", val3)
    print("val4", val4)
    print("val5", val5)
    print("val6", val6)

    aaes = (0.5 * val1) + (0.5 * val2)
    aaet = (0.5 * val3) + (0.5 * val4)

    nnaar={
        "aaes": aaes,
        "aaet": aaet,
        "val11": val11,
        "val22": val22,
        "val33": val33,
        "val44": val44,
        "val55": val55,
        "val66": val66,
    }
    return nnaar



def compute_privacy(train, val, syn):

    L_SAMPLE = min(
        len(train),
        len(val),
        len(syn),
    ) *0 + 1000

    # add subject_id
    for i, p in enumerate(train):
        p["subject_id"] = i
    for i, p in enumerate(val):
        p["subject_id"] = i
    for i, p in enumerate(syn):
        p["subject_id"] = i
    nnaar = calc_nnaar(random.sample(train,L_SAMPLE),
                        random.sample(val, L_SAMPLE), 
                        random.sample(syn, L_SAMPLE))




    return nnaar


def compute_llm_metrics(train, val, syn,pad_token_id=0):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.metrics import mean_squared_error
    from math import sqrt


    # TF-IDF
    D_flattened = [' '.join([str(x) for x in patient]) for patient in train]
    G_flattened = [' '.join([str(x) for x in patient]) for patient in syn]
    V_flattened = [' '.join([str(x) for x in patient]) for patient in val]

    # Initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()

    # Fit the vectorizer on D and transform both D and G
    tfidf_D = vectorizer.fit_transform(D_flattened)
    tfidf_G = vectorizer.transform(G_flattened)
    tfidf_V = vectorizer.transform(V_flattened)

    # Compute cosine similarity between the mean TF-IDF vectors of D and G
    mean_tfidf_D = np.asarray(tfidf_D.mean(axis=0))
    mean_tfidf_G = np.asarray(tfidf_G.mean(axis=0))
    mean_tfidf_V = np.asarray(tfidf_V.mean(axis=0))


    # Rouge-bleu metrics
    from datasets import load_metric
    import evaluate
    # Load metrics
    bleu_metric = load_metric("bleu",trust_remote_code=True)
    rouge_metric = load_metric("rouge",trust_remote_code=True)
    XXX = 1000000
    # rouge
    ref_train = [str([(g) for g in gen if g != pad_token_id])[1:-1] for gen in train[:XXX]]
    hyp_eval = [str([(r) for r in ref if r != pad_token_id])[1:-1] for ref in val[:XXX]]
    hyp_syn = [str([(r) for r in ref if r != pad_token_id])[1:-1] for ref in syn[:XXX]]

    r_score_eval = rouge_metric.compute(predictions=hyp_eval, references=ref_train)
    r_score_syn = rouge_metric.compute(predictions=hyp_syn, references=ref_train)

    # bleu
    ref_train = [([(g) for g in gen if g != pad_token_id]) for gen in train[:XXX]]
    hyp_eval = [([(r) for r in ref if r != pad_token_id]) for ref in val[:XXX]]
    hyp_syn = [([(r) for r in ref if r != pad_token_id]) for ref in syn[:XXX]]

    b_score_eval = bleu_metric.compute(predictions=hyp_eval, references=[[ref] for ref in ref_train])
    b_score_syn = bleu_metric.compute(predictions=hyp_syn, references=[[ref] for ref in ref_train])

    return {
        'tfidf-rmse-syn': sqrt(mean_squared_error(mean_tfidf_D.flatten(), mean_tfidf_G.flatten())),
        'tfidf-rmse-val': sqrt(mean_squared_error(mean_tfidf_D.flatten(), mean_tfidf_V.flatten())),
        'rouge1-val': r_score_eval['rouge1'].mid.fmeasure,
        'rouge1-syn': r_score_syn['rouge1'].mid.fmeasure,
        'rouge2-val': r_score_eval['rouge2'].mid.fmeasure,
        'rouge2-syn': r_score_syn['rouge2'].mid.fmeasure,
        'rougeL-val': r_score_eval['rougeL'].mid.fmeasure,
        'rougeL-syn': r_score_syn['rougeL'].mid.fmeasure,
        'bleu-val': b_score_eval['bleu'],
        'bleu-syn': b_score_syn['bleu'],
    }

def compute_ngram_metrics(train_ehr_dataset, val_ehr_dataset, synthetic_ehr_dataset, pad_token_id, L_SAMPLE):
    th=1
    for n in tqdm([1, 2, 3, 4], desc="n-gram"):
        probs_train = dataset_ngram(
            random.sample(train_ehr_dataset, L_SAMPLE),
            n=n,
            th=[0, th],
            batch_size=2048,
            pad_token_id=pad_token_id,
        )

        probs_val = dataset_ngram(
            random.sample(val_ehr_dataset, L_SAMPLE),
            n=n,
            th=[0, th],
            batch_size=2048,
            pad_token_id=pad_token_id,
        )

        probs_gen = dataset_ngram(
            random.sample(synthetic_ehr_dataset, L_SAMPLE),
            n=n,
            th=[0, 1],
            batch_size=2048,
            pad_token_id=pad_token_id,
        )

        fig = plot_scatter(probs_val, probs_gen, log_scale=False, title=f"{n}-gram")
        wandb.log({f"{n}-gram/val-gen": wandb.Plotly(fig)})
        fig = plot_scatter(probs_train, probs_gen, log_scale=False, title=f"{n}-gram")
        wandb.log({f"{n}-gram/train-gen": wandb.Plotly(fig)})
        fig = plot_scatter(probs_train, probs_val, log_scale=False, title=f"{n}-gram")
        wandb.log({f"{n}-gram/train-val": wandb.Plotly(fig)})

        fig = plot_scatter(probs_val, probs_gen, log_scale=True, title=f"{n}-gram")
        wandb.log({f"{n}-gram-log/val-gen": wandb.Plotly(fig)})
        fig = plot_scatter(probs_train, probs_gen, log_scale=True, title=f"{n}-gram")
        wandb.log({f"{n}-gram-log/train-gen": wandb.Plotly(fig)})
        fig = plot_scatter(probs_train, probs_val, log_scale=True, title=f"{n}-gram")
        wandb.log({f"{n}-gram-log/train-val": wandb.Plotly(fig)})

        # report metrics

        metric = compute_metrics(probs_train, probs_val)
        wandb.log({(f"train-val-n{n}/" + k): v for k, v in metric.items()})

        metric = compute_metrics(probs_train, probs_gen)
        wandb.log({(f"train-gen-n{n}/" + k): v for k, v in metric.items()})



def plot_tsne(REAL: torch.Tensor, FAKE: torch.Tensor, N: int = 10000) -> go.Figure:
    # for t-SNE
    # third party
    from MulticoreTSNE import MulticoreTSNE as TSNE

    X = np.concatenate([REAL[:N], FAKE[:N]], axis=0)  # [2*bs, hidden_dim]

    tsne = TSNE(n_components=2, perplexity=30, learning_rate=10, n_jobs=4)

    X_tsne = tsne.fit_transform(X)  # [2*bs, 2]

    N2 = int(X.shape[0] / 2)

    fig_tsne = go.Figure()
    _ = fig_tsne.add_trace(
        go.Scatter(x=X_tsne[:N2, 0], y=X_tsne[:N2, 1], mode="markers", name="real")
    )
    _ = fig_tsne.add_trace(
        go.Scatter(x=X_tsne[N2:, 0], y=X_tsne[N2:, 1], mode="markers", name="fake")
    )

    # fig_tsne.show()

    # # plot histogram
    # REAL2 = REAL[:,3].flatten()
    # FAKE2 = FAKE[:,3].flatten()
    # fig_hist = go.Figure()
    # _ = fig_hist.add_trace(go.Histogram(x=REAL2, nbinsx=100, name='real'))
    # _ = fig_hist.add_trace(go.Histogram(x=FAKE2, nbinsx=100, name='fake'))
    # fig_hist.show()

    return fig_tsne



def plot_tsne2(data:dict, N: int = 10000) -> go.Figure:
    # for t-SNE
    # third party
    from MulticoreTSNE import MulticoreTSNE as TSNE

    X = []
    for k in data.keys():
        # random sample
        random_indices = np.random.choice(data[k].shape[0], size=N, replace=False)
        X.append( data[k][random_indices, :] )

    X = np.concatenate(X, axis=0)  # [2*bs, hidden_dim]

    tsne = TSNE(n_components=2, perplexity=30, learning_rate=10, n_jobs=4)

    X_tsne = tsne.fit_transform(X)  # [2*bs, 2]

    # N2 = int(X.shape[0] / 2)

    fig_tsne = go.Figure()

    for k in data.keys():
        # print(X_tsne.shape)
        
        _ = fig_tsne.add_trace(
            go.Scatter(x=X_tsne[:N, 0], y=X_tsne[:N, 1], mode="markers", name=k)
        )
        # set opacity for each name




        X_tsne = X_tsne[N:]
    
    # set opacity
    fig_tsne.update_traces(marker=dict(opacity=0.75, size=5))

    # set marker border width
    # fig_tsne.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))

    # fig_tsne.show()

    # # plot histogram
    # REAL2 = REAL[:,3].flatten()
    # FAKE2 = FAKE[:,3].flatten()
    # fig_hist = go.Figure()
    # _ = fig_hist.add_trace(go.Histogram(x=REAL2, nbinsx=100, name='real'))
    # _ = fig_hist.add_trace(go.Histogram(x=FAKE2, nbinsx=100, name='fake'))
    # fig_hist.show()

    return fig_tsne




def plot_corr(
    df_train_real: pd.DataFrame,
    df_train_fake: pd.DataFrame,
    df_test: pd.DataFrame,
    state_vars: List,
    corr_method: str = "",
    corr_th: float = 0.2,
) -> Any:
    def impute(df: pd.DataFrame) -> pd.DataFrame:

        # if CORR_METHOD=='ffill':
        df = df.copy()
        df[state_vars] = (
            df[["id"] + state_vars]
            .groupby("id")
            .fillna(method="ffill", limit=6)
        )

        # # mean imputation
        # for col in state_vars:
        #     df[col].fillna(df[col].mean(), inplace=True)

        return df

    def corr_agg(df: pd.DataFrame) -> np.ndarray:

        df = impute(df)
        temp = df.groupby("RecordID")[state_vars].corr()
        grouped = [group.droplevel(0).values for _, group in temp.groupby("RecordID")]

        corr = np.stack(grouped, axis=0)
        corr.shape

        # corr[np.abs(corr)<0.2]=0
        # set np.nan to zero
        corr[np.isnan(corr)] = 0

        # set upper triangle and diagonal to zero
        for i in range(corr.shape[0]):
            corr[i][np.triu_indices_from(corr[i], k=0)] = 0

        corr = corr.mean(0)

        return corr

    def compute_temp_corr(
        mat_true: np.ndarray, mat_syn: np.ndarray, th: float = 0.0
    ) -> float:

        norm_const = mat_true.shape[0] * (mat_true.shape[0] - 1) / 2
        mat_true = mat_true.copy()
        mat_syn = mat_syn.copy()

        mat_true[np.abs(mat_true) < th] = 0
        mat_syn[np.abs(mat_syn) < th] = 0

        # set upper triangle and diagonal to zero
        mat_true[np.triu_indices_from(mat_true, k=0)] = 0
        mat_syn[np.triu_indices_from(mat_syn, k=0)] = 0

        # set nans to zero
        mat_true[np.isnan(mat_true)] = 0
        mat_syn[np.isnan(mat_syn)] = 0

        x = np.mean((mat_true - mat_syn) ** 2)

        # compute L1 loss
        x = np.sum(np.abs(mat_true - mat_syn)) / norm_const

        # # compute frobenius norm
        # x = np.linalg.norm(mat_true-mat_syn)

        return x

    if corr_method == "ffill":

        corr_train = impute(df_train_real)[state_vars].corr(min_periods=100).values
        corr_val = impute(df_test)[state_vars].corr(min_periods=100).values
        corr_gen = impute(df_train_fake)[state_vars].corr(min_periods=100).values
    elif corr_method == "agg":
        corr_train = corr_agg(df_train_real)
        corr_val = corr_agg(df_test)
        corr_gen = corr_agg(df_train_fake)
    else:
        corr_train = df_train_real[state_vars].corr(min_periods=100).values
        corr_val = df_test[state_vars].corr(min_periods=100).values
        corr_gen = df_train_fake[state_vars].corr(min_periods=100).values

    # # print MSE of correlation matrices
    # corr_train[np.abs(corr_train)<corr_th]=0
    # corr_val[np.abs(corr_val)<corr_th]=0
    # corr_gen[np.abs(corr_gen)<corr_th]=0


    # confusion matrix for correlation
    print("[info] computing confusion matrix for correlation")
    corr_train2 = corr_train.copy()
    corr_val2 = corr_val.copy()
    corr_gen2 = corr_gen.copy()

    # set upper triangle to 100 (including diagonal)
    corr_train2[np.triu_indices_from(corr_train2, k=0)] = 100
    corr_val2[np.triu_indices_from(corr_val2, k=0)] = 100
    corr_gen2[np.triu_indices_from(corr_gen2, k=0)] = 100

    # set nans to 100
    corr_train2[np.isnan(corr_train2)] = 100
    corr_val2[np.isnan(corr_val2)] = 100
    corr_gen2[np.isnan(corr_gen2)] = 100

    corr_range = {
        'high-neg':[-1,-0.5],
        'medium-neg':[-0.5,-0.2],
        'low': [-0.2,0.2],
        'medium-pos':[0.2,0.5],
        'high-pos':[0.5,1],
    }

    for i,(k,v) in enumerate(corr_range.items()):
        print(k)

        corr_train2[np.logical_and(corr_train>=v[0],corr_train<=v[1])]=i
        corr_val2[np.logical_and(corr_val>=v[0],corr_val<=v[1])]=i
        corr_gen2[np.logical_and(corr_gen>=v[0],corr_gen<=v[1])]=i

        # compute the accuracy for class i
        y_train = (corr_train2==i)        
        y_train = y_train[np.tril_indices_from(y_train, k=-1)]

        y_val = (corr_val2==i)
        y_val = y_val[np.tril_indices_from(y_val, k=-1)]

        y_gen = (corr_gen2==i)
        y_gen = y_gen[np.tril_indices_from(y_gen, k=-1)]

        print((corr_train2==i).sum(), (corr_val2==i).sum(), (corr_gen2==i).sum())
        # acc for train-val
        acc = (y_train==y_val).sum()/len(y_train)

        # acc for train-gen
        acc2 = (y_train==y_gen).sum()/len(y_train)

        print(f"acc-train-val: {acc:.3f}")
        print(f"acc-train-gen: {acc2:.3f}")

        # f1 score
        from sklearn.metrics import f1_score
        f1 = f1_score(y_train,y_val)
        f1_2 = f1_score(y_train,y_gen)

        print(f"f1-train-val: {f1:.3f}")
        print(f"f1-train-gen: {f1_2:.3f}")

    # only keep lower triangle (not diagonal), then flatten
    corr_train2 = corr_train2[np.tril_indices(corr_train2.shape[0], -1)]
    corr_val2 = corr_val2[np.tril_indices(corr_val2.shape[0], -1)]
    corr_gen2 = corr_gen2[np.tril_indices(corr_gen2.shape[0], -1)]
    
    print((corr_train2==0).sum(), (corr_val2==0).sum(), (corr_gen2==0).sum())


    # plot
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt

    
    new_labels = [0,1,2,3,4,100]

    label_map = {
        0: 'high-neg',
        1: 'medium-neg',
        2: 'low',
        3: 'medium-pos',
        4: 'high-pos',
        100: 'nan'
    }

    
    cm_train_val = confusion_matrix(corr_train2, corr_val2, labels=new_labels)
    cm_train_gen = confusion_matrix(corr_train2, corr_gen2, labels=new_labels)


    disp_train_val = ConfusionMatrixDisplay(confusion_matrix=cm_train_val, display_labels=[label_map[i] for i in new_labels])
    disp_train_val.plot(xticks_rotation=45)
    plt.title('Correlation Confusion Matrix - Training and Validation')


    disp_train_gen = ConfusionMatrixDisplay(confusion_matrix=cm_train_gen, display_labels=[label_map[i] for i in new_labels])
    disp_train_gen.plot(xticks_rotation=45)
    plt.title('Correlation Confusion Matrix - Training and Generated')


    # save to wandb
    if wandb.run:
        wandb.log({
            'corr-conf/train_val': disp_train_val.figure_,
            'corr-conf/train_gen': disp_train_gen.figure_
            
            })
        wandb.log({
            'corr-conf-cm/train_val': cm_train_val,
            'corr-conf-cm/train_gen':  cm_train_gen
        })


    # return corr_train2, corr_val2, corr_gen2

    metric = {
        "corr-diff/Train-Synthetic": compute_temp_corr(corr_train, corr_gen, th=corr_th),
        "corr-diff/Train-Test": compute_temp_corr(corr_train, corr_val, th=corr_th),
    }
    print(metric)

    mask = np.logical_or(np.abs(corr_train) < corr_th, np.abs(corr_val) < corr_th)
    mask = np.abs(corr_train) < corr_th
    

    # set upper triangle and diagonal to zero
    corr_train[np.triu_indices_from(corr_train, k=0)] = np.nan
    corr_val[np.triu_indices_from(corr_val, k=0)] = np.nan
    corr_gen[np.triu_indices_from(corr_gen, k=0)] = np.nan

    # set values below th to nan
    corr_train[np.abs(corr_train) < corr_th] = np.nan
    corr_val[np.abs(corr_val) < corr_th] = np.nan
    corr_gen[np.abs(corr_gen) < corr_th] = np.nan

    y_names = state_vars.copy()
    y_names.reverse()

    mask_nan_gen = np.isnan(corr_gen)

    corr_diff = np.abs(corr_gen - corr_train)  # /(corr_train+1e-9)
    # set nans to zero
    corr_diff[mask] = 0
    # corr_diff[np.isnan(corr_diff)]=-1
    corr_diff[mask_nan_gen] = 0
    corr_diff[np.abs(corr_diff) < corr_th] = 0

    # mirror horizontally (if using plotly go)
    corr_train = corr_train[::-1]
    corr_val = corr_val[::-1]
    corr_gen = corr_gen[::-1]
    corr_diff = corr_diff[::-1]

    # plot all in a subplot 2 by 2
    # each subplot should be square
    fig = make_subplots(rows=2, cols=2, subplot_titles=("Train", "Val", "Gen", "Diff"))

    sub_train = go.Heatmap(
        z=corr_train, x=state_vars, y=y_names, colorscale="RdBu", zmin=-1, zmax=1
    )
    _ = fig.add_trace(sub_train, row=1, col=1)

    sub_val = go.Heatmap(
        z=corr_val, x=state_vars, y=y_names, colorscale="RdBu", zmin=-1, zmax=1
    )
    _ = fig.add_trace(sub_val, row=1, col=2)

    sub_gen = go.Heatmap(
        z=corr_gen, x=state_vars, y=y_names, colorscale="RdBu", zmin=-1, zmax=1
    )
    _ = fig.add_trace(sub_gen, row=2, col=1)

    sub_diff = go.Heatmap(
        z=corr_diff, x=state_vars, y=y_names, colorscale="RdBu", zmin=-1, zmax=1
    )
    # _ = fig.add_trace(sub_diff, row=2, col=2)

    # # set hiehgts and width of each subplot to be equal
    # _ = fig.update_layout(height=1200, width=1100, title_text="Correlation matrices")

    #     _ = fig.update_layout(
    #     # subplot_titles=['Heatmap 1', 'Heatmap 2', 'Heatmap 3', 'Heatmap 4'],
    #     # grid=dict(rows=2, columns=2),  # 2x2 grid for 4 subplots
    #     row_heights=[1, 1],  # Set the relative heights (1:1 ratio for both rows)
    #     column_widths=[1, 1]  # Set the relative widths (1:1 ratio for both columns)
    # )

    for i in range(1, 5):
        _ = fig.update_yaxes(scaleanchor=f"x{str(i)}", scaleratio=1)
        _ = fig.update_xaxes(scaleanchor=f"y{str(i)}", scaleratio=1)

    # if wandb
    if wandb.run:
        wandb.log(metric)
        wandb.log({"corr-mat": fig})
    return fig, (sub_train, sub_val, sub_gen, sub_diff)




def plot_corr3(
    df_train_real: pd.DataFrame,
    df_test: pd.DataFrame,
    state_vars: List,
    corr_method: str = "",
    corr_th: float = 0.2,
) -> Any:
    def impute(df: pd.DataFrame) -> pd.DataFrame:

        # if CORR_METHOD=='ffill':
        df = df.copy()
        df[state_vars] = (
            df[["id"] + state_vars]
            .groupby("id")
            .fillna(method="ffill", limit=6)
        )

        # # mean imputation
        # for col in state_vars:
        #     df[col].fillna(df[col].mean(), inplace=True)

        return df

    def corr_agg(df: pd.DataFrame) -> np.ndarray:

        df = impute(df)
        temp = df.groupby("RecordID")[state_vars].corr()
        grouped = [group.droplevel(0).values for _, group in temp.groupby("RecordID")]

        corr = np.stack(grouped, axis=0)
        corr.shape

        # corr[np.abs(corr)<0.2]=0
        # set np.nan to zero
        corr[np.isnan(corr)] = 0

        # set upper triangle and diagonal to zero
        for i in range(corr.shape[0]):
            corr[i][np.triu_indices_from(corr[i], k=0)] = 0

        corr = corr.mean(0)

        return corr

    def compute_temp_corr(
        mat_true: np.ndarray, mat_syn: np.ndarray, th: float = 0.0
    ) -> float:

        norm_const = mat_true.shape[0] * (mat_true.shape[0] - 1) / 2
        mat_true = mat_true.copy()
        mat_syn = mat_syn.copy()

        mat_true[np.abs(mat_true) < th] = 0
        mat_syn[np.abs(mat_syn) < th] = 0

        # set upper triangle and diagonal to zero
        mat_true[np.triu_indices_from(mat_true, k=0)] = 0
        mat_syn[np.triu_indices_from(mat_syn, k=0)] = 0

        # set nans to zero
        mat_true[np.isnan(mat_true)] = 0
        mat_syn[np.isnan(mat_syn)] = 0

        x = np.mean((mat_true - mat_syn) ** 2)

        # compute L1 loss
        x = np.sum(np.abs(mat_true - mat_syn)) / norm_const

        # # compute frobenius norm
        # x = np.linalg.norm(mat_true-mat_syn)

        return x

    if corr_method == "ffill":

        corr_train = impute(df_train_real)[state_vars].corr(min_periods=100).values
        corr_val = impute(df_test)[state_vars].corr(min_periods=100).values
    elif corr_method == "agg":
        corr_train = corr_agg(df_train_real)
        corr_val = corr_agg(df_test)
    else:
        corr_train = df_train_real[state_vars].corr(min_periods=100).values
        corr_val = df_test[state_vars].corr(min_periods=100).values

    # # print MSE of correlation matrices
    # corr_train[np.abs(corr_train)<corr_th]=0
    # corr_val[np.abs(corr_val)<corr_th]=0
    # corr_gen[np.abs(corr_gen)<corr_th]=0


    # confusion matrix for correlation
    print("[info] computing confusion matrix for correlation")
    corr_train2 = corr_train.copy()
    corr_val2 = corr_val.copy()

    # set upper triangle to 100 (including diagonal)
    corr_train2[np.triu_indices_from(corr_train2, k=0)] = 100
    corr_val2[np.triu_indices_from(corr_val2, k=0)] = 100

    # set nans to 100
    corr_train2[np.isnan(corr_train2)] = 100
    corr_val2[np.isnan(corr_val2)] = 100

    corr_range = {
        'high-neg':[-1,-0.5],
        'medium-neg':[-0.5,-0.2],
        'low': [-0.2,0.2],
        'medium-pos':[0.2,0.5],
        'high-pos':[0.5,1],
    }

    for i,(k,v) in enumerate(corr_range.items()):
        # print(k)

        corr_train2[np.logical_and(corr_train>=v[0],corr_train<=v[1])]=i
        corr_val2[np.logical_and(corr_val>=v[0],corr_val<=v[1])]=i

        # compute the accuracy for class i
        y_train = (corr_train2==i)        
        y_train = y_train[np.tril_indices_from(y_train, k=-1)]

        y_val = (corr_val2==i)
        y_val = y_val[np.tril_indices_from(y_val, k=-1)]

        # # print((corr_train2==i).sum(), (corr_val2==i).sum(), (corr_gen2==i).sum())
        # acc for train-val
        acc = (y_train==y_val).sum()/len(y_train)

        # # acc for train-gen
        # acc2 = (y_train==y_gen).sum()/len(y_train)

        # print(f"acc-train-val: {acc:.3f}")
        # # print(f"acc-train-gen: {acc2:.3f}")

        # f1 score
        from sklearn.metrics import f1_score
        f1 = f1_score(y_train,y_val)
        # f1_2 = f1_score(y_train,y_gen)

        # print(f"f1-train-val: {f1:.3f}")
        # print(f"f1-train-gen: {f1_2:.3f}")

    # only keep lower triangle (not diagonal), then flatten
    corr_train2 = corr_train2[np.tril_indices(corr_train2.shape[0], -1)]
    corr_val2 = corr_val2[np.tril_indices(corr_val2.shape[0], -1)]
    
    # print((corr_train2==0).sum(), (corr_val2==0).sum(), (corr_gen2==0).sum())


    # plot
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt

    
    new_labels = [0,1,2,3,4]

    label_map = {
        0: 'high-neg',
        1: 'medium-neg',
        2: 'low',
        3: 'medium-pos',
        4: 'high-pos',
        100: 'nan'
    }

    label_map = {
        0: '0',
        1: '1',
        2: '2',
        3: '3',
        4: '4',
        100: 'nan'
    }
    cm_train_val = confusion_matrix(corr_train2, corr_val2, labels=new_labels)
    # cm_train_gen = confusion_matrix(corr_train2, corr_gen2, labels=new_labels)



    # Convert size from cm to inches
    width_cm = 5  # Desired width in cm
    height_cm = 5  # Desired height in cm
    width_inch = width_cm / 2.54
    height_inch = height_cm / 2.54

    # Set the figure size in inches
    plt.figure(figsize=(width_inch, height_inch))
    # Set font size globally
    plt.rcParams.update({'font.size': 24})  # Change 16 to your preferred font size


    disp_train_val = ConfusionMatrixDisplay(confusion_matrix=cm_train_val, display_labels=[label_map[i] for i in new_labels])
    disp_train_val.plot(xticks_rotation=45)
    # plt.title('Correlation Confusion Matrix - Training and Validation')


    # disp_train_gen = ConfusionMatrixDisplay(confusion_matrix=cm_train_gen, display_labels=[label_map[i] for i in new_labels])
    # disp_train_gen.plot(xticks_rotation=45)
    # plt.title('Correlation Confusion Matrix - Training and Generated')


    # # save to wandb
    # if wandb.run:
    #     wandb.log({
    #         'corr-conf/train_val': disp_train_val.figure_,
    #         'corr-conf/train_gen': disp_train_gen.figure_
            
    #         })
    #     wandb.log({
    #         'corr-conf-cm/train_val': cm_train_val,
    #         'corr-conf-cm/train_gen':  cm_train_gen
    #     })


    # return corr_train2, corr_val2, corr_gen2

    metric_TDC = compute_temp_corr(corr_train, corr_val, th=corr_th)
    
    return cm_train_val,metric_TDC
    print(metric)

    mask = np.logical_or(np.abs(corr_train) < corr_th, np.abs(corr_val) < corr_th)
    mask = np.abs(corr_train) < corr_th
    

    # set upper triangle and diagonal to zero
    corr_train[np.triu_indices_from(corr_train, k=0)] = np.nan
    corr_val[np.triu_indices_from(corr_val, k=0)] = np.nan
    corr_gen[np.triu_indices_from(corr_gen, k=0)] = np.nan

    # set values below th to nan
    corr_train[np.abs(corr_train) < corr_th] = np.nan
    corr_val[np.abs(corr_val) < corr_th] = np.nan
    corr_gen[np.abs(corr_gen) < corr_th] = np.nan

    y_names = state_vars.copy()
    y_names.reverse()

    mask_nan_gen = np.isnan(corr_gen)

    corr_diff = np.abs(corr_gen - corr_train)  # /(corr_train+1e-9)
    # set nans to zero
    corr_diff[mask] = 0
    # corr_diff[np.isnan(corr_diff)]=-1
    corr_diff[mask_nan_gen] = 0
    corr_diff[np.abs(corr_diff) < corr_th] = 0

    # mirror horizontally (if using plotly go)
    corr_train = corr_train[::-1]
    corr_val = corr_val[::-1]
    corr_gen = corr_gen[::-1]
    corr_diff = corr_diff[::-1]

    # plot all in a subplot 2 by 2
    # each subplot should be square
    fig = make_subplots(rows=2, cols=2, subplot_titles=("Train", "Val", "Gen", "Diff"))

    sub_train = go.Heatmap(
        z=corr_train, x=state_vars, y=y_names, colorscale="RdBu", zmin=-1, zmax=1
    )
    _ = fig.add_trace(sub_train, row=1, col=1)

    sub_val = go.Heatmap(
        z=corr_val, x=state_vars, y=y_names, colorscale="RdBu", zmin=-1, zmax=1
    )
    _ = fig.add_trace(sub_val, row=1, col=2)

    sub_gen = go.Heatmap(
        z=corr_gen, x=state_vars, y=y_names, colorscale="RdBu", zmin=-1, zmax=1
    )
    _ = fig.add_trace(sub_gen, row=2, col=1)

    sub_diff = go.Heatmap(
        z=corr_diff, x=state_vars, y=y_names, colorscale="RdBu", zmin=-1, zmax=1
    )
    # _ = fig.add_trace(sub_diff, row=2, col=2)

    # # set hiehgts and width of each subplot to be equal
    # _ = fig.update_layout(height=1200, width=1100, title_text="Correlation matrices")

    #     _ = fig.update_layout(
    #     # subplot_titles=['Heatmap 1', 'Heatmap 2', 'Heatmap 3', 'Heatmap 4'],
    #     # grid=dict(rows=2, columns=2),  # 2x2 grid for 4 subplots
    #     row_heights=[1, 1],  # Set the relative heights (1:1 ratio for both rows)
    #     column_widths=[1, 1]  # Set the relative widths (1:1 ratio for both columns)
    # )

    for i in range(1, 5):
        _ = fig.update_yaxes(scaleanchor=f"x{str(i)}", scaleratio=1)
        _ = fig.update_xaxes(scaleanchor=f"y{str(i)}", scaleratio=1)

    # if wandb
    if wandb.run:
        wandb.log(metric)
        wandb.log({"corr-mat": fig})
    return fig, (sub_train, sub_val, sub_gen, sub_diff)




def plot_corr2(
    df_train_real: pd.DataFrame,
    df_train_fake: pd.DataFrame,
    df_test: pd.DataFrame,
    state_vars: List,
    corr_method: str = "",
    corr_th: float = 0.2,
) -> Any:
    def impute(df: pd.DataFrame) -> pd.DataFrame:

        # if CORR_METHOD=='ffill':
        df = df.copy()
        df[state_vars] = (
            df[["id"] + state_vars]
            .groupby("id")
            .fillna(method="ffill", limit=6)
        )

        # # mean imputation
        # for col in state_vars:
        #     df[col].fillna(df[col].mean(), inplace=True)

        return df

    def corr_agg(df: pd.DataFrame) -> np.ndarray:

        df = impute(df)
        temp = df.groupby("RecordID")[state_vars].corr()
        grouped = [group.droplevel(0).values for _, group in temp.groupby("RecordID")]

        corr = np.stack(grouped, axis=0)
        corr.shape

        # corr[np.abs(corr)<0.2]=0
        # set np.nan to zero
        corr[np.isnan(corr)] = 0

        # set upper triangle and diagonal to zero
        for i in range(corr.shape[0]):
            corr[i][np.triu_indices_from(corr[i], k=0)] = 0

        corr = corr.mean(0)

        return corr

    def compute_temp_corr(
        mat_true: np.ndarray, mat_syn: np.ndarray, th: float = 0.2
    ) -> float:
        print(th)
        norm_const = mat_true.shape[0] * (mat_true.shape[0] - 1) / 2
        mat_true = mat_true.copy()
        mat_syn = mat_syn.copy()

        mat_true[np.abs(mat_true) < th] = 0
        mat_syn[np.abs(mat_syn) < th] = 0

        # set upper triangle and diagonal to zero
        mat_true[np.triu_indices_from(mat_true, k=0)] = 0
        mat_syn[np.triu_indices_from(mat_syn, k=0)] = 0

        # set nans to zero
        mat_true[np.isnan(mat_true)] = 0
        mat_syn[np.isnan(mat_syn)] = 0

        x = np.mean((mat_true - mat_syn) ** 2)

        # compute L1 loss
        x = np.sum(np.abs(mat_true - mat_syn)) / norm_const

        # # compute frobenius norm
        # x = np.linalg.norm(mat_true-mat_syn)

        return x

    if corr_method == "ffill":

        corr_train = impute(df_train_real)[state_vars].corr(min_periods=100).values
        corr_val = impute(df_test)[state_vars].corr(min_periods=100).values
        corr_gen = impute(df_train_fake)[state_vars].corr(min_periods=100).values
    elif corr_method == "agg":
        corr_train = corr_agg(df_train_real)
        corr_val = corr_agg(df_test)
        corr_gen = corr_agg(df_train_fake)
    else:
        corr_train = df_train_real[state_vars].corr(min_periods=100).values
        corr_val = df_test[state_vars].corr(min_periods=100).values
        corr_gen = df_train_fake[state_vars].corr(min_periods=100).values

    # set upper triangle to 100 (including diagonal)
    corr_train[np.triu_indices_from(corr_train, k=0)] = 100
    corr_val[np.triu_indices_from(corr_val, k=0)] = 100
    corr_gen[np.triu_indices_from(corr_gen, k=0)] = 100

    # set nans to 100
    corr_train[np.isnan(corr_train)] = 100
    corr_val[np.isnan(corr_val)] = 100
    corr_gen[np.isnan(corr_gen)] = 100

    corr_range = {
        'high-neg':[-1,-0.5],
        'medium-neg':[-0.5,-0.2],
        'low': [-0.2,0.2],
        'medium-pos':[0.2,0.5],
        'high-pos':[0.5,1],
    }
    
    corr_train2 = corr_train.copy()
    corr_val2 = corr_val.copy()
    corr_gen2 = corr_gen.copy()
    for i,(k,v) in enumerate(corr_range.items()):
        print(k)

        corr_train2[np.logical_and(corr_train>=v[0],corr_train<=v[1])]=i
        corr_val2[np.logical_and(corr_val>=v[0],corr_val<=v[1])]=i
        corr_gen2[np.logical_and(corr_gen>=v[0],corr_gen<=v[1])]=i

        # compute the accuracy for class i
        y_train = (corr_train2==i)        
        y_train = y_train[np.tril_indices_from(y_train, k=-1)]

        y_val = (corr_val2==i)
        y_val = y_val[np.tril_indices_from(y_val, k=-1)]

        y_gen = (corr_gen2==i)
        y_gen = y_gen[np.tril_indices_from(y_gen, k=-1)]

        print((corr_train2==i).sum(), (corr_val2==i).sum(), (corr_gen2==i).sum())
        # acc for train-val
        acc = (y_train==y_val).sum()/len(y_train)

        # acc for train-gen
        acc2 = (y_train==y_gen).sum()/len(y_train)

        print(f"acc-train-val: {acc:.3f}")
        print(f"acc-train-gen: {acc2:.3f}")

        # f1 score
        from sklearn.metrics import f1_score
        f1 = f1_score(y_train,y_val)
        f1_2 = f1_score(y_train,y_gen)

        print(f"f1-train-val: {f1:.3f}")
        print(f"f1-train-gen: {f1_2:.3f}")

    # corr_train2 = np.nan_to_num(corr_train2, nan=2)
    # corr_val2 = np.nan_to_num(corr_val2, nan=2)
    # corr_gen2 = np.nan_to_num(corr_gen2, nan=2)

    # only keep lower triangle (not diagonal), then flatten
    corr_train2 = corr_train2[np.tril_indices(corr_train2.shape[0], -1)]
    corr_val2 = corr_val2[np.tril_indices(corr_val2.shape[0], -1)]
    corr_gen2 = corr_gen2[np.tril_indices(corr_gen2.shape[0], -1)]
    
    print((corr_train2==0).sum(), (corr_val2==0).sum(), (corr_gen2==0).sum())

    return corr_train2, corr_val2, corr_gen2

    # compute the confusion matrix for corr_train and corr_val
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(corr_train[np.tril_indices_from(corr_train, k=-1)],corr_val[np.tril_indices_from(corr_val, k=-1)])
    cm2 = confusion_matrix(corr_train[np.tril_indices_from(corr_train, k=-1)],corr_gen[np.tril_indices_from(corr_gen, k=-1)])


    # plot the confusion matrix using plotly
    fig = go.Figure(data=go.Heatmap(z=cm, x=range(5), y=range(5)))
    fig.show()

    fig = go.Figure(data=go.Heatmap(z=cm2, x=range(5), y=range(5)))
    fig.show()
    

        



    # print(
    #     f"MSE of correlation matrices: {compute_temp_corr(corr_train[:,:], corr_gen[:,:],th=corr_th):.3f}"
    # )
    # print(
    #     f"MSE of correlation matrices BL: {compute_temp_corr(corr_train[:,:], corr_val[:,:],th=corr_th):.3f}"
    # )

    # metric = {
    #     "TCD[Train-Synthetic]/stats.tc_corr": compute_temp_corr(corr_train, corr_gen),
    #     "TCD[Train-Test]/stats.tc_corr": compute_temp_corr(corr_train, corr_val),
    # }

    # mask = np.logical_or(np.abs(corr_train) < corr_th, np.abs(corr_val) < corr_th)
    # mask = np.abs(corr_train) < corr_th
    

    # set upper triangle and diagonal to zero
    corr_train[np.triu_indices_from(corr_train, k=0)] = 2
    corr_val[np.triu_indices_from(corr_val, k=0)] = 2
    corr_gen[np.triu_indices_from(corr_gen, k=0)] = 2
    y_names = state_vars.copy()
    y_names.reverse()

    mask_nan_gen = np.isnan(corr_gen)

    corr_diff = np.abs(corr_gen - corr_train)  # /(corr_train+1e-9)
    # set nans to zero
    # corr_diff[mask] = 0
    # corr_diff[np.isnan(corr_diff)]=-1
    # corr_diff[mask_nan_gen] = 0
    # corr_diff[np.abs(corr_diff) < corr_th] = 0

    # mirror horizontally (if using plotly go)
    corr_train = corr_train[::-1]
    corr_val = corr_val[::-1]
    corr_gen = corr_gen[::-1]
    corr_diff = corr_diff[::-1]

    # plot all in a subplot 2 by 2
    # each subplot should be square
    colorscale = [
        [0, 'rgb(165,0,38)'],    # Color for value 0
        [1, 'rgb(244,109,67)'],  # Color for value 1
        [2, 'rgb(253,174,97)'],   # Color for value 2
        [3, 'rgb(116,173,209)'], # Color for value 3
        [4, 'rgb(49,54,149)']     # Color for value 4
    ]
    colorscale = 'RdBu'
    fig = make_subplots(rows=2, cols=2, subplot_titles=("Train", "Val", "Gen", "Diff"))

    sub_train = go.Heatmap(
        z=corr_train, x=state_vars, y=y_names, colorscale=colorscale, zmin=0, zmax=4, zmid=2
    )
    _ = fig.add_trace(sub_train, row=1, col=1)

    sub_val = go.Heatmap(
        z=corr_val, x=state_vars, y=y_names, colorscale=colorscale, zmin=0, zmax=4, zmid=2
    )
    _ = fig.add_trace(sub_val, row=1, col=2)

    sub_gen = go.Heatmap(
        z=corr_gen, x=state_vars, y=y_names, colorscale=colorscale, zmin=0, zmax=4, zmid=2
    )
    _ = fig.add_trace(sub_gen, row=2, col=1)

    sub_diff = go.Heatmap(
        z=corr_diff, x=state_vars, y=y_names, colorscale=colorscale, zmin=0, zmax=4, zmid=2
    )
    _ = fig.add_trace(sub_diff, row=2, col=2)

    # set hiehgts and width of each subplot to be equal
    _ = fig.update_layout(height=1200, width=1100, title_text="Correlation matrices")

    #     _ = fig.update_layout(
    #     # subplot_titles=['Heatmap 1', 'Heatmap 2', 'Heatmap 3', 'Heatmap 4'],
    #     # grid=dict(rows=2, columns=2),  # 2x2 grid for 4 subplots
    #     row_heights=[1, 1],  # Set the relative heights (1:1 ratio for both rows)
    #     column_widths=[1, 1]  # Set the relative widths (1:1 ratio for both columns)
    # )

    for i in range(1, 5):
        _ = fig.update_yaxes(scaleanchor=f"x{str(i)}", scaleratio=1)
        _ = fig.update_xaxes(scaleanchor=f"y{str(i)}", scaleratio=1)

    # if wandb
    if wandb.run:
        wandb.log(metric)
        wandb.log({"Correlation Matrices": fig})
    return fig, (sub_train, sub_val, sub_gen, sub_diff)





def compute_corr_mat(
    all_values: torch.Tensor,
    all_masks: torch.Tensor,
    NVARS: int = 40,
    method: str = "none",
) -> np.ndarray:
    # all_values.shape = (*, n_vars) in [-1,1]
    # all_masks.shape = (*, n_vars) -1: missing, 1: not missing
    # method = 'none' or 'ffill'

    IMG_HEIGHT = all_values.shape[-2]
    NUM_SAMPLES = all_values.shape[0]

    # define a dataframe
    df = pd.DataFrame(all_values[:, :, :, :NVARS].reshape(-1, NVARS))
    df[(all_masks[:, :, :, :NVARS] < 0).reshape(-1, NVARS)] = np.nan

    if method == "ffill":
        # forward fill with limit of 6
        df["id"] = np.repeat(np.arange(NUM_SAMPLES), IMG_HEIGHT)
        # 0 0 0 0 ... 0 (L) | ....| 100 100 100 (L)
        df = df.groupby("id").ffill(limit=6)

    corr_mat = df.corr().values
    corr_mat[np.abs(corr_mat) < 0.2] = 0
    corr_mat[np.triu_indices_from(corr_mat)] = 0

    return corr_mat



def mat2img(mat: torch.Tensor) -> np.ndarray:
    X = mat.cpu().detach().numpy()

    # def grayscale_to_blue_red(x):
    #     if x < 0.5:
    #         return (int(255 * (1 - x)), int(255 * x), int(255 * x))
    #     else:
    #         return (int(255 * (1 - x)), int(255 * (1 - x)), int(255 * x))

    red_values = (255 * (1 - X[0])).astype(int)  # shape (H, W)
    blue_values = (255 * X[0]).astype(int)  # shape (H, W)
    green_values = (255 * X[0]).astype(int)  # shape (H, W)
    green_values[X[0] >= 0.5] = (255 * (1 - X[0, X[0] >= 0.5])).astype(int)
    # Apply the color mapping to the grayscale image

    # colored_image = np.apply_along_axis(grayscale_to_blue_red, 0, X[[0],:,:])
    colored_image = np.stack(
        [red_values, green_values, blue_values], axis=0
    )  # shape (3, H, W)

    colored_image = np.uint8(colored_image)
    colored_image[:, X[1, :, :] < 0.5] = 255

    return colored_image


def save_examples(
    real: torch.Tensor, fake: torch.Tensor, n_ts: int = 40, epoch_no: int = -1
) -> None:
    # everythinig is between -1 and 1

    NVARS = n_ts  # number of time series variables
    L_CUT = NVARS + 1
    N_samples = 9

    images = []

    for i in range(N_samples):

        img_grid_real = torchvision.utils.make_grid(
            real[i, :, :, :L_CUT], normalize=True, nrow=1
        )  # .unsqueeze(1) # shape (n_channels, H, W)
        img_grid_fake = torchvision.utils.make_grid(
            fake[i, :, :, :L_CUT], normalize=True, nrow=1
        )  # .unsqueeze(1) # shape (n_channels, H, W)

        upscaled_img_grid_real = torch.nn.functional.interpolate(
            img_grid_real.unsqueeze(0), scale_factor=4, mode="nearest"
        ).squeeze(
            0
        )  # shape (n_channels, 4H, 4W)
        upscaled_img_grid_fake = torch.nn.functional.interpolate(
            img_grid_fake.unsqueeze(0), scale_factor=4, mode="nearest"
        ).squeeze(
            0
        )  # shape (n_channels, 4H, 4W)

        real_image = mat2img(upscaled_img_grid_real)  # [3, 4H, 4W]
        fake_image = mat2img(upscaled_img_grid_fake)  # [3, 4H, 4W]

        # for the fake_image, set the white pixels to #fadf93
        def set_white_to_gray(image_tensor: np.ndarray) -> np.ndarray:
            # Check where all three channels are 255 (white)
            white_mask = np.all(image_tensor == 255, axis=0)

            # Set these locations to gray (128, 128, 128)
            # Update the Red, Green, and Blue channels 252, 238, 197
            image_tensor[0, white_mask] = 252  # Red
            image_tensor[1, white_mask] = 238  # Green
            image_tensor[2, white_mask] = 197  # Blue

            return image_tensor

        fake_image = set_white_to_gray(fake_image)

        # add a watermark to fake image with gray
        def add_watermark(
            image_tensor: np.ndarray,
            text: str = "Watermark",
            position: Any = "center",
            font_size: int = 40,
            font_color: Tuple = (128, 128, 128),
        ) -> np.ndarray:
            # third party
            from PIL import Image, ImageDraw, ImageFont

            # Convert the tensor to a PIL image
            image = Image.fromarray(
                np.transpose(image_tensor, (1, 2, 0)).astype("uint8")
            )

            # Create an ImageDraw object
            draw = ImageDraw.Draw(image)

            # Load a font
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except IOError:
                font = ImageFont.load_default()

            # Calculate the text position for center alignment using textbbox
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            if position == "center":
                W, H = image.size
                x = (W - text_width) / 2
                y = (H - text_height) / 2
            else:
                x, y = position

            # Add text to image
            draw.text((x, y), text, font=font, fill=font_color)

            # Convert back to a numpy array if needed
            result_tensor = np.array(image).transpose(2, 0, 1)
            return result_tensor

        fake_image = add_watermark(fake_image, text="Synthetic")

        # add a black border on top, bottom and left of real image
        real_image = np.pad(
            real_image, ((0, 0), (2, 2), (2, 0)), "constant", constant_values=0
        )
        # add a gray border on right of real image

        real_image = np.pad(
            real_image, ((0, 0), (0, 0), (0, 1)), "constant", constant_values=128
        )

        # add a black border on top, bottom and right of fake image
        fake_image = np.pad(
            fake_image, ((0, 0), (2, 2), (0, 2)), "constant", constant_values=0
        )
        # add a gray border on left of fake image
        fake_image = np.pad(
            fake_image, ((0, 0), (0, 0), (1, 0)), "constant", constant_values=196
        )

        real_fake_image = np.concatenate(
            [real_image, fake_image], axis=2
        )  # [3, 4H, 8W]

        images.append(real_fake_image)

    # we have 9 real images and 9 fake images
    # we want to make a 3x6 grid
    # we want to alternate real and fake images:
    # example first row: real1, fake1, real2, fake2, real3, fake3

    tot_H = images[0].shape[1] * 3
    tot_W = images[0].shape[2] * 3

    X = np.concatenate(images, axis=2).reshape(3, tot_H, tot_W)
    X = np.concatenate(
        [
            np.concatenate([images[0], images[1], images[2]], axis=2),
            np.concatenate([images[3], images[4], images[5]], axis=2),
            np.concatenate([images[6], images[7], images[8]], axis=2),
        ],
        axis=1,
    )

    # # map to RGB
    # third party
    from PIL import Image

    # Convert the NumPy array to a PIL Image
    pil_image = Image.fromarray(X.transpose(1, 2, 0))

    # pil_image.save("output_image.png")

    if wandb.run is not None:
        if epoch_no == -1:  # during evaluation
            wandb.log(
                {
                    "Real-vs-Synthetic/Images": [
                        wandb.Image(pil_image, caption="Real vs Synthetic")
                    ]
                }
            )
        else:  # when called during training
            wandb.log(
                {
                    "Real-vs-Synthetic/Images": [
                        wandb.Image(pil_image, caption="Real vs Synthetic")
                    ]
                },
                step=epoch_no,
                commit=False,
            )

    return pil_image




def handle_cgan(
    df_filt: pd.DataFrame,
    state_vars: List,
    granularity: int = 1,
    target_dim: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    dyn = []

    grouped = df_filt.groupby("RecordID")

    # Create a list of DataFrames
    dyn = [
        group[["Time"] + state_vars].rename(columns={"Time": "time"})
        for _, group in grouped
    ]

    all_times = []
    all_dyn = []
    all_masks = []

    for SAMPLE in dyn:
        # time = SAMPLE.time.values
        # times_padded = pd.DataFrame({'time':np.arange(0,   int(max(time)),granularity)})

        # print(time,len(time),max(time))
        # print(times_padded.values.flatten(),len(times_padded.values.flatten()))

        # create a np array from 0 to max(time) with granularity 0.5 ()including max(time)) use linspace
        time_padded = np.linspace(
            0, int(target_dim * granularity) - granularity, target_dim
        )  # shape is (target_dim,)

        df_time_padded = pd.DataFrame({"time": time_padded})
        temp = df_time_padded.merge(
            SAMPLE, how="outer", on="time"
        )  # .sort_values(by='time_padded').

        dyn_padded = temp[state_vars].fillna(0).values
        mask_padded = temp[state_vars].notnull().astype(int).values

        all_times.append(torch.from_numpy(time_padded)[:target_dim])
        all_dyn.append(torch.from_numpy(dyn_padded)[:target_dim])
        all_masks.append(torch.from_numpy(mask_padded)[:target_dim])
        # print(time_padded.shape,dyn_padded.shape,mask_padded.shape)

    # return all_masks,all_dyn,all_times

    all_masks = torch.stack(all_masks, dim=0)
    all_dyn = torch.stack(all_dyn, dim=0)
    all_times = torch.stack(all_times, dim=0)

    # PADDING
    # target_dim = TARGET_DIM
    padding_needed = target_dim - len(state_vars)

    all_masks_padded = (
        torch.nn.functional.pad(all_masks, (0, padding_needed)).unsqueeze(1).float()
    )  # add channel dim
    all_dyn_padded = (
        torch.nn.functional.pad(all_dyn, (0, padding_needed)).unsqueeze(1).float()
    )  # add channel dim
    # all_times_padded = torch.nn.functional.pad(all_times, (0, padding_needed))
    all_masks_padded.shape
    all_dyn_padded.shape

    all_data = torch.stack([all_dyn_padded, all_masks_padded], dim=1)
    all_data.shape

    # SCALE all_data to [-1,1]
    all_masks_padded = all_masks_padded * 2 - 1
    all_dyn_padded = all_dyn_padded * 2 - 1
    all_dyn_padded[all_masks_padded < 0] = 0
    # all_data = all_data*2-1
    # all_data[:,0,:,:][all_data[:,1,:,:]<0]=0

    return all_masks_padded.float(), all_dyn_padded.float()#, all_sta.float()



def prepro(
    df: pd.DataFrame, df_mean: pd.DataFrame, df_std: pd.DataFrame, state_vars: List
) -> pd.DataFrame:

    df = df.copy()
    # # missingness rate old
    # df_missing = df[['RecordID']+state_vars].groupby('RecordID').apply(lambda x:x.isnull().sum()/x.shape[0])[state_vars].reset_index()

    # missingness rate (normalized by max time)
    # df_missing = (
    #     df[["id"] + ["Hours"] + state_vars]
    #     .groupby("id")
    #     .apply(lambda x: x[state_vars].isnull().sum() / max(x.Hours))[state_vars]
    #     .reset_index()
    # )

    # # change column names to "mr+{var}"
    # df_missing.columns = ["id"] + ["mr_" + var for var in state_vars]

    # mean imputation for missing values
    df[state_vars] = df[state_vars].fillna(df[state_vars].mean())
    # print(df[state_vars].isnull().sum().sum())  
    # standardize
    df[state_vars] = (df[state_vars] - df_mean) / df_std

    # print('after stdzation',df[state_vars].isnull().sum().sum())  
    
    # group by id and compute statistics (min.max,mean,std) for each variable

    df2 = df.groupby("id")[state_vars].agg(["min", "max", "mean", "std"]).reset_index()

    # print('after agg',df2.isnull().sum().sum(),np.isinf(df2).sum().sum()  )
    # combine the names first and second level column indices

    df2.columns = ["_".join(col).strip() for col in df2.columns.values]

    df2.rename(columns={"id_": "id"}, inplace=True)

    # df2 = df2.merge(df[["id", "outcome"]].drop_duplicates(), on=["id"], how="inner")

    # df2 = df2.merge(df_missing, on=["id"], how="inner")
    print('after add mr',df2.isnull().sum().sum(), np.isinf(df2).sum().sum())  

    # assert df2.isnull().sum().sum()==0, f"Null values in df2: {df2.isnull().sum().sum()}"
    # assert np.isinf(df2).sum().sum()==0, f"Inf values in df2: {np.isinf(df2).sum().sum()}"
    return df2



def genTSembeddings(
    df: pd.DataFrame, df_static: pd.DataFrame, state_vars: List,col_labels:List, df_base: pd.DataFrame = None
) -> Tuple[pd.DataFrame, pd.Series]:

    if df_base is not None:
        df_mean, df_std = df_base[state_vars].mean(), df_base[state_vars].std()
    else:
        df_mean, df_std = df[state_vars].mean(), df[state_vars].std()

    print("[info] Preprocessing ")
    df_pro = prepro(df, df_mean, df_std, state_vars)

    features_names = df_pro.columns.tolist()
    features_names.remove("id")
    # features_names.remove("outcome")

    if df_pro.id.nunique() != df_static.id.nunique():
        print("Different number of unique ids in df and df_static")
        print(df_pro.id.nunique(), df_static.id.nunique())
        print("Taking intersection of ids")

        ids = set(df_pro.id.unique()).intersection(set(df_static.id.unique()))
        df_pro = df_pro[df_pro.id.isin(ids)]
        df_static = df_static[df_static.id.isin(ids)]

    X = df_pro[features_names].reset_index(drop=True)
    # y = df_pro["outcome"]
    y = df_static[col_labels].reset_index(drop=True)
    X
    assert X.shape[0] == y.shape[0]
    return X, y



def compute_utility(
    X_train_real,
    X_train_fake,
    X_test_real,
    X_test_fake,
    y_train_real,
    y_train_fake,
    y_test_real,
    y_test_fake,
):
    from sklearn.metrics import (
        accuracy_score,
        roc_auc_score,
        f1_score,
        average_precision_score,
        roc_curve,
        precision_recall_curve,
        precision_score,
        recall_score,
        confusion_matrix,
    )
    from sklearn.multiclass import OneVsRestClassifier
    import lightgbm as lgb


    UTILITY_MODE = ["TRTR", "TSTR"]
    metrics = dict()
    if "TRTR" in UTILITY_MODE:
        # TRTR

        # # compute scale_pos_weight
        # scale_pos_weight = (y_train_real == 0).sum() / (y_train_real == 1).sum()

        print("\n[info] TRTR")

        # fit real model
        model_real = lgb.LGBMClassifier(
            random_state=42,verbosity=-1
        )

        if y_train_real.shape[1]>1:
            print('multi-label classifer')
            model_real = OneVsRestClassifier(model_real)

        model_real.fit(X_train_real, y_train_real)

        # eval model
        y_pred = model_real.predict(X_test_real)
        y_score = model_real.predict_proba(X_test_real)
        y_true = y_test_real.values

        from sklearn.metrics import hamming_loss, jaccard_score, f1_score

        print("Hamming Loss:", hamming_loss(y_true, y_pred))
        print("Jaccard Score:", jaccard_score(y_true, y_pred, average='samples'))
        print("F1 Score (micro):", f1_score(y_true, y_pred, average='micro'))
        print("F1 Score (macro):", f1_score(y_true, y_pred, average='macro'))


        print(f"AUROC: {roc_auc_score(y_true, y_score, multi_class='ovr')}")
        print(f"AUPRC: {average_precision_score(y_true, y_score, average='micro')}")
        # print(f"AUPRC: {average_precision_score(y_true, y_score)}")
        # print(f"F1: {f1_score(y_true, y_pred)}")

        metrics.update(
            {
                "TRTR/AUROC": roc_auc_score(y_true, y_score, multi_class='ovr'),
                "TRTR/AUPRC": average_precision_score(y_true, y_score, average='micro'),
                "TRTR/F1-micro": f1_score(y_true, y_pred, average='micro'),
                "TRTR/F1-macro": f1_score(y_true, y_pred, average='macro'),
                "TRTR/Hamming Loss": hamming_loss(y_true, y_pred),
                "TRTR/Jaccard Score": jaccard_score(y_true, y_pred, average='samples'),

                
            }
        )

    if "TSTR" in UTILITY_MODE:

        # TSTR
        print("\n[info] TSTR")

        # fit real model
        model_real = lgb.LGBMClassifier(
            random_state=42, verbosity=-1
        )

        if y_train_real.shape[1]>1:
            print('multi-label classifer')
            model_real = OneVsRestClassifier(model_real)

        model_real.fit(X_train_real, y_train_real, )

        # eval model
        y_pred = model_real.predict(X_test_fake)
        y_score = model_real.predict_proba(X_test_fake)
        y_true = y_test_fake.values

        from sklearn.metrics import hamming_loss, jaccard_score, f1_score

        print("Hamming Loss:", hamming_loss(y_true, y_pred))
        print("Jaccard Score:", jaccard_score(y_true, y_pred, average='samples'))
        print("F1 Score (micro):", f1_score(y_true, y_pred, average='micro'))
        print("F1 Score (macro):", f1_score(y_true, y_pred, average='macro'))


        print(f"AUROC: {roc_auc_score(y_true, y_score, multi_class='ovr')}")
        print(f"AUPRC: {average_precision_score(y_true, y_score, average='micro')}")
        # print(f"AUPRC: {average_precision_score(y_true, y_score)}")
        # print(f"F1: {f1_score(y_true, y_pred)}")

        metrics.update(
            {
                "TSTR/AUROC": roc_auc_score(y_true, y_score, multi_class='ovr'),
                "TSTR/AUPRC": average_precision_score(y_true, y_score, average='micro'),
                "TSTR/F1-micro": f1_score(y_true, y_pred, average='micro'),
                "TSTR/F1-macro": f1_score(y_true, y_pred, average='macro'),
                "TSTR/Hamming Loss": hamming_loss(y_true, y_pred),
                "TSTR/Jaccard Score": jaccard_score(y_true, y_pred, average='samples'),
            }
        )
        

    return metrics


def compute_utility2(
    X_train, y_train, X_test, y_test, X_fake, y_fake,
    train_fake_ratio=[(1,0),(0,1)]
):
    from sklearn.metrics import (
        accuracy_score,
        roc_auc_score,
        f1_score,
        average_precision_score,
        roc_curve,
        precision_recall_curve,
        precision_score,
        recall_score,
        confusion_matrix,
        hamming_loss, jaccard_score, f1_score
    )
    from sklearn.multiclass import OneVsRestClassifier
    import lightgbm as lgb

    

    def plot_metrics():
        df_metrics = pd.DataFrame(metrics).reset_index()
        # Reset the index to bring the 'AUROC-ovr', etc. into columns

        # Melt the DataFrame
        df_melted = pd.melt(df_metrics, id_vars='index')

        # Rename the columns for better clarity (optional)
        df_melted.columns = ['Metric', 'Column_Index_1', 'Column_Index_2', 'Value']

        # If you want the columns to have meaningful names (if possible)
        df_melted = df_melted.sort_values(by=['Metric']).reset_index(drop=True)

        # rename the index columns
        df_melted.rename(columns={'Column_Index_1': 'train_ratio', 'Column_Index_2': 'syn_ratio'}, inplace=True)

        df_melted.sort_values(by=['Metric','train_ratio','syn_ratio'], ascending=[False,True, False])


        # px.bar(df_melted, x='train_ratio', y='Value', color='syn_ratio')

        for metric in df_melted['Metric'].unique():
            # metric
            fig = go.Figure()
            df_melted_metric = df_melted[df_melted['Metric'] == metric].sort_values(by=['train_ratio','syn_ratio'], ascending=[True, True])
            # df_melted_metric
            
            for syn_ratio in df_melted_metric['syn_ratio'].unique():
                if syn_ratio == 1:
                    args = {'name': '+ synthetic',
                            
                            # dashed line
                            'line': dict(color='blue'),
                            

                            }
                else:
                    args = {'name': f'No synthetic',
                            'line': dict(dash='dash', color='red'),
                            }
                temp = df_melted_metric[df_melted_metric['syn_ratio'] == syn_ratio]

                _ = fig.add_trace(go.Scatter(x=temp['train_ratio'], y=temp['Value'], **args))

            _ = fig.update_layout(barmode='group', title=metric)
            _ = fig.update_traces(opacity=0.75)
            # set x ticks
            _ = fig.update_xaxes(tickvals=temp['train_ratio'].unique())
            # set x label
            _ = fig.update_xaxes(title_text='train_ratio')
            # set y grid size to 0.01
            _ = fig.update_yaxes(dtick=0.01)

            if wandb.run:
                wandb.log({f'AUG-Plots/{metric}': fig})
            else:
                fig.show()
    


    multi_label = False

    metrics = dict()


    for real_ratio, fake_ratio in tqdm(train_fake_ratio, total=len(train_fake_ratio), leave=False):
        X_train_frac = X_train.sample(frac=real_ratio, random_state=42)
        y_train_frac = y_train.loc[X_train_frac.index]

        X_fake_frac = X_fake.sample(frac=fake_ratio, random_state=42)
        y_fake_frac = y_fake.loc[X_fake_frac.index]

        X = pd.concat([X_train_frac, X_fake_frac])
        y = pd.concat([y_train_frac, y_fake_frac])

        print(X_train_frac.shape, X_fake_frac.shape, X.shape)
        print(y_train_frac.shape, y_fake_frac.shape, y.shape)
        if y.shape[1]>1:
            print('multi-label classifer')
            multi_label = True
            model = OneVsRestClassifier(lgb.LGBMClassifier(random_state=42,  verbosity=1))

        else:
            print('binary classifer')
            model = lgb.LGBMClassifier(random_state=42,  verbosity=1)
            y=y.values.flatten()
        print(X.shape, y.shape)
        model.fit(X, y)

        # eval model
        y_pred = model.predict(X_test) # 
        y_score = model.predict_proba(X_test)
        y_true = y_test.values

        print(y_pred.shape, y_score.shape, y_true.shape)

        if multi_label:
            metrics.update({
                (real_ratio, fake_ratio):
                {
                    "AUROC-ovr": roc_auc_score(y_true, y_score, multi_class='ovr'),
                    "AUROC-micro": roc_auc_score(y_true, y_score, average='micro'),
                    "AUPRC-micro": average_precision_score(y_true, y_score, average='micro'),
                    "AUPRC-macro": average_precision_score(y_true, y_score, average='macro'),
                    "F1-micro": f1_score(y_true, y_pred, average='micro'),
                    "F1-macro": f1_score(y_true, y_pred, average='macro'),
                    # "Hamming Loss": hamming_loss(y_true, y_pred),
                    # "Jaccard Score": jaccard_score(y_true, y_pred, average='samples'),

                    
                }}
            )
        else:
            y_score = y_score[:,1]
            y_true = y_true[:,0]

            metrics.update({
                (real_ratio, fake_ratio):
                {
                    "AUROC": roc_auc_score(y_true, y_score),
                    "AUPRC": average_precision_score(y_true, y_score),
                    "F1": f1_score(y_true, y_pred),
                    
                    
                           }}
            )


    if wandb.run:

        wandb.log( {f"AUG/{k}": v for k, v in metrics.items()})

        if (1,0) in metrics:
            wandb.log({
                f'TRTR/{k}':v for k,v in metrics[(1,0)].items()
            })
        
        if (0,1) in metrics:
            wandb.log({
                f'TSTR/{k}':v for k,v in metrics[(0,1)].items()
            })

    # log to wandb or plot
    plot_metrics()

    return metrics



def compute_temp_corr(
    df_train_real: pd.DataFrame,
    df_train_fake: pd.DataFrame,
    df_test: pd.DataFrame,
    state_vars: List,
    corr_method: str = "",
    corr_th: float = 0.2,
) -> dict:
    def impute(df: pd.DataFrame) -> pd.DataFrame:

        # if CORR_METHOD=='ffill':
        df = df.copy()
        df[state_vars] = (
            df[["id"] + state_vars].groupby("id").fillna(method="ffill", limit=6)
        )

        # # mean imputation
        # for col in state_vars:
        #     df[col].fillna(df[col].mean(), inplace=True)

        return df

    def corr_agg(df: pd.DataFrame) -> np.ndarray:

        df = impute(df)
        temp = df.groupby("id")[state_vars].corr()
        grouped = [group.droplevel(0).values for _, group in temp.groupby("id")]

        corr = np.stack(grouped, axis=0)
        corr.shape

        # corr[np.abs(corr)<0.2]=0
        # set np.nan to zero
        corr[np.isnan(corr)] = 0

        # set upper triangle and diagonal to zero
        for i in range(corr.shape[0]):
            corr[i][np.triu_indices_from(corr[i], k=0)] = 0

        corr = corr.mean(0)

        return corr

    def compute_metric(
        mat_true: np.ndarray, mat_syn: np.ndarray, th: float = 0.2
    ) -> float:
        norm_const = mat_true.shape[0] * (mat_true.shape[0] - 1) / 2

        mat_true2 = mat_true.copy()
        mat_syn2 = mat_syn.copy()

        mat_true2[np.abs(mat_true2) < th] = 0
        mat_syn2[np.abs(mat_syn2) < th] = 0

        # set upper triangle and diagonal to zero
        mat_true2[np.triu_indices_from(mat_true2, k=0)] = 0
        mat_syn2[np.triu_indices_from(mat_syn2, k=0)] = 0

        # set nans to zero
        mat_true2[np.isnan(mat_true2)] = 0
        mat_syn2[np.isnan(mat_syn2)] = 0

        x = np.mean((mat_true2 - mat_syn2) ** 2)

        # compute frobenius norm
        x = np.linalg.norm(mat_true2 - mat_syn2)

        # compute L1 loss
        x = np.sum(np.abs(mat_true2 - mat_syn2)) / norm_const

        return x

    if corr_method == "ffill":

        corr_train = impute(df_train_real)[state_vars].corr().values
        corr_val = impute(df_test)[state_vars].corr().values
        corr_gen = impute(df_train_fake)[state_vars].corr().values
    elif corr_method == "agg":
        corr_train = corr_agg(df_train_real)
        corr_val = corr_agg(df_test)
        corr_gen = corr_agg(df_train_fake)
    else:
        corr_train = df_train_real[state_vars].corr().values
        corr_val = df_test[state_vars].corr().values
        corr_gen = df_train_fake[state_vars].corr().values

    # print MSE of correlation matrices

    # print(f"MSE of correlation matrices: {compute_metric(corr_train, corr_gen):.3f}")
    # print(f"MSE of correlation matrices BL: {compute_metric(corr_train, corr_val):.3f}")

    metric = {
        "TCD[Train-Synthetic]/stats.tc_corr": compute_metric(corr_train, corr_gen),
        "TCD[Train-Test]/stats.tc_corr": compute_metric(corr_train, corr_val),
    }

    # log to wandb
    # mirror horizontally (if using plotly go)
    mat_true = corr_train.copy()
    mat_syn = corr_gen.copy()
    th = corr_th

    mat_true[np.abs(mat_true) < th] = 0
    mat_syn[np.abs(mat_syn) < th] = 0

    # set upper triangle and diagonal to zero
    mat_true[np.triu_indices_from(mat_true, k=0)] = 0
    mat_syn[np.triu_indices_from(mat_syn, k=0)] = 0

    # set nans to zero
    mat_true[np.isnan(mat_true)] = 0
    mat_syn[np.isnan(mat_syn)] = 0

    # mirror horizontally (if using plotly go)
    mat_true = mat_true[::-1]
    mat_syn = mat_syn[::-1]

    y_names = state_vars.copy()
    y_names.reverse()
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Real", "Synthetic"))

    sub_train = go.Heatmap(
        z=mat_true, x=state_vars, y=y_names, colorscale="RdBu", zmin=-1, zmax=1
    )
    _ = fig.add_trace(sub_train, row=1, col=1)

    sub_val = go.Heatmap(
        z=mat_syn, x=state_vars, y=y_names, colorscale="RdBu", zmin=-1, zmax=1
    )
    _ = fig.add_trace(sub_val, row=1, col=2)
    wandb.log({"Correlation Matrices": fig})

    return metric




def compute_synthcity2(X_real, X_fake):
    # import sys
    # sys.path.insert(0, "/mlodata1/hokarami/synthcity/src")
    from synthcity.plugins.core.dataloader import GenericDataLoader
    from synthcity.metrics import Metrics
    from synthcity.metrics.scores import ScoreEvaluator

    from pathlib import Path


    scores = ScoreEvaluator()
    X = GenericDataLoader(
        X_real,
        # target_column="label_ihm",
    )

    X_syn = GenericDataLoader(
        X_fake,
        # target_column="label_ihm",
    )
    X_ref_syn = X_syn
    X_augmented = None
    selected_metrics = {
        "privacy": [
            "delta-presence",
            "k-anonymization",
            "k-map",
            "distinct l-diversity",
            "identifiability_score",
            "DomiasMIA_BNAF",
            "DomiasMIA_KDE",
            "DomiasMIA_prior",
        ]
    }
    selected_metrics = {
        "stats": [
            "alpha_precision",
        ],
        "privacy": ["identifiability_score"],
    }
    # selected_metrics={
    #     'sanity': ['data_mismatch', 'common_rows_proportion', 'nearest_syn_neighbor_distance', 'close_values_probability', 'distant_values_probability'],
    #             'stats': ['jensenshannon_dist', 'chi_squared_test', 'feature_corr', 'inv_kl_divergence', 'ks_test', 'max_mean_discrepancy', 'wasserstein_dist', 'prdc', 'alpha_precision', 'survival_km_distance'],
    #             'performance': ['linear_model', 'mlp', 'xgb', 'feat_rank_distance'],
    #             'detection': ['detection_xgb', 'detection_mlp', 'detection_gmm', 'detection_linear'],
    #     'privacy': ['delta-presence', 'k-anonymization', 'k-map', 'distinct l-diversity', 'identifiability_score']}

    selected_metrics = {
        "stats": [
            "alpha_precision",
            # 'jensenshannon_dist', 'chi_squared_test', 'feature_corr', 'inv_kl_divergence', 'ks_test', 'max_mean_discrepancy',
            "prdc",
            # 'survival_km_distance',
            # 'wasserstein_dist'
        ],
        #         'stats': [
        #             'alpha_precision',
        #             'jensenshannon_dist', 'chi_squared_test', 'feature_corr', 'inv_kl_divergence', 'ks_test', 'max_mean_discrepancy', 'wasserstein_dist', 'prdc', 'alpha_precision', 'survival_km_distance'
        #         ],
        "privacy": [
            #  'delta-presence', 'k-anonymization', 'k-map', 'distinct l-diversity',
            "identifiability_score"
        ],
    }

    selected_metrics = {
        "stats": [
            # 'alpha_precision',
            "prdc",
        ],
        "privacy": ["identifiability_score"],
    }

    n_repeats = 1

    # # DEBUG
    # scores = ScoreEvaluator()
    # from sklearn.datasets import load_diabetes
    # X, y = load_diabetes(return_X_y=True, as_frame=True)
    # X["target"] = y

    # loader = GenericDataLoader(
    #     X,
    #     target_column="target",
    #     sensitive_columns=["sex"],
    # )

    # X = loader.train()
    # X_test = loader.test()
    # X_syn = loader
    # X_ref_syn = loader
    # X_augmented = None
    print("\n[info] computing synthcity metrics X_train and X_syn")
    scores = ScoreEvaluator()

    for _ in range(n_repeats):
        evaluation = Metrics.evaluate(
            X,
            X_syn,
            X,
            X_ref_syn,
            X_augmented,
            metrics=selected_metrics,
            task_type="classification",
            workspace=Path("workspace"),
            use_cache=False,
        )
        mean_score = evaluation["mean"].to_dict()
        errors = evaluation["errors"].to_dict()
        duration = evaluation["durations"].to_dict()
        direction = evaluation["direction"].to_dict()

        for key in mean_score:
            scores.add(
                key,
                mean_score[key],
                errors[key],
                duration[key],
                direction[key],
            )
    metrics_syn = scores.to_dataframe()["mean"].to_dict()
    # metrics_syn_std = scores.to_dataframe()["std"].to_dict()


    return metrics_syn#, metrics_syn_std



def compute_synthcity(X_real, X_fake, X_test):
    # import sys
    # sys.path.insert(0, "/mlodata1/hokarami/synthcity/src")
    from synthcity.plugins.core.dataloader import GenericDataLoader
    from synthcity.metrics import Metrics
    from synthcity.metrics.scores import ScoreEvaluator

    from pathlib import Path


    scores = ScoreEvaluator()
    X = GenericDataLoader(
        X_real,
        # target_column="outcome",
    )
    X_test = GenericDataLoader(
        X_test,
        # target_column="outcome",
    )
    X_syn = GenericDataLoader(
        X_fake,
        # target_column="outcome",
    )
    X_ref_syn = X_syn
    X_augmented = None
    selected_metrics = {
        "privacy": [
            "delta-presence",
            "k-anonymization",
            "k-map",
            "distinct l-diversity",
            "identifiability_score",
            "DomiasMIA_BNAF",
            "DomiasMIA_KDE",
            "DomiasMIA_prior",
        ]
    }
    selected_metrics = {
        "stats": [
            "alpha_precision",
        ],
        "privacy": ["identifiability_score"],
    }
    # selected_metrics={
    #     'sanity': ['data_mismatch', 'common_rows_proportion', 'nearest_syn_neighbor_distance', 'close_values_probability', 'distant_values_probability'],
    #             'stats': ['jensenshannon_dist', 'chi_squared_test', 'feature_corr', 'inv_kl_divergence', 'ks_test', 'max_mean_discrepancy', 'wasserstein_dist', 'prdc', 'alpha_precision', 'survival_km_distance'],
    #             'performance': ['linear_model', 'mlp', 'xgb', 'feat_rank_distance'],
    #             'detection': ['detection_xgb', 'detection_mlp', 'detection_gmm', 'detection_linear'],
    #     'privacy': ['delta-presence', 'k-anonymization', 'k-map', 'distinct l-diversity', 'identifiability_score']}

    selected_metrics = {
        "stats": [
            "alpha_precision",
            # 'jensenshannon_dist', 'chi_squared_test', 'feature_corr', 'inv_kl_divergence', 'ks_test', 'max_mean_discrepancy',
            "prdc",
            # 'survival_km_distance',
            # 'wasserstein_dist'
        ],
        #         'stats': [
        #             'alpha_precision',
        #             'jensenshannon_dist', 'chi_squared_test', 'feature_corr', 'inv_kl_divergence', 'ks_test', 'max_mean_discrepancy', 'wasserstein_dist', 'prdc', 'alpha_precision', 'survival_km_distance'
        #         ],
        "privacy": [
            #  'delta-presence', 'k-anonymization', 'k-map', 'distinct l-diversity',
            "identifiability_score"
        ],
    }

    selected_metrics = {
        "stats": [
            # 'alpha_precision',
            "prdc",
        ],
        "privacy": ["identifiability_score"],
    }

    n_repeats = 1

    # # DEBUG
    # scores = ScoreEvaluator()
    # from sklearn.datasets import load_diabetes
    # X, y = load_diabetes(return_X_y=True, as_frame=True)
    # X["target"] = y

    # loader = GenericDataLoader(
    #     X,
    #     target_column="target",
    #     sensitive_columns=["sex"],
    # )

    # X = loader.train()
    # X_test = loader.test()
    # X_syn = loader
    # X_ref_syn = loader
    # X_augmented = None

    print("\n[info] computing synthcity metrics X_train and X_test")
    scores = ScoreEvaluator()

    for _ in range(n_repeats):
        evaluation = Metrics.evaluate(
            X,
            X_test,
            X,
            X_ref_syn,
            X_augmented,
            metrics=selected_metrics,
            task_type="classification",
            workspace=Path("workspace"),
            use_cache=False,
        )
        mean_score = evaluation["mean"].to_dict()
        errors = evaluation["errors"].to_dict()
        duration = evaluation["durations"].to_dict()
        direction = evaluation["direction"].to_dict()

        for key in mean_score:
            scores.add(
                key,
                mean_score[key],
                errors[key],
                duration[key],
                direction[key],
            )
    metrics_syn_test = scores.to_dataframe()["mean"].to_dict()

    metrics_syn_test = {
        ("Synthcity[Train-Test]/" + k): v for k, v in metrics_syn_test.items()
    }

    print("\n[info] computing synthcity metrics X_train and X_syn")
    scores = ScoreEvaluator()

    for _ in range(n_repeats):
        evaluation = Metrics.evaluate(
            X,
            X_syn,
            X,
            X_ref_syn,
            X_augmented,
            metrics=selected_metrics,
            task_type="classification",
            workspace=Path("workspace"),
            use_cache=False,
        )
        mean_score = evaluation["mean"].to_dict()
        errors = evaluation["errors"].to_dict()
        duration = evaluation["durations"].to_dict()
        direction = evaluation["direction"].to_dict()

        for key in mean_score:
            scores.add(
                key,
                mean_score[key],
                errors[key],
                duration[key],
                direction[key],
            )
    metrics_syn = scores.to_dataframe()["mean"].to_dict()

    metrics_syn = {
        ("Synthcity[Train-Synthetic]/" + k): v for k, v in metrics_syn.items()
    }


    # combine two metrics
    metrics = {**metrics_syn_test, **metrics_syn}

    # print(metrics)

    return metrics


def compute_nnaa(REAL, FAKE, TEST):

    def ff(A, B, self=False):
        # print(A.shape, B.shape)
        np.sum(A**2, axis=1).reshape(A.shape[0], 1).shape, np.sum(B.T**2, axis=0).shape
        a = np.sum(A**2, axis=1).reshape(A.shape[0], 1) + np.sum(
            B.T**2, axis=0
        ).reshape(1, B.shape[0])
        b = np.dot(A, B.T) * 2
        distance_matrix = a - b
        a.shape, b.shape, distance_matrix.shape
        np.min(distance_matrix, axis=0)
        if self == True:
            np.fill_diagonal(distance_matrix, np.inf)
        # print(distance_matrix[:5,:5])
        # print(np.min(distance_matrix[:5,:5], axis=1))
        min_dist_AB = np.min(distance_matrix, axis=1)
        min_dist_BA = np.min(distance_matrix, axis=0)

        return min_dist_AB, min_dist_BA

    distance_TT, _ = ff(REAL, REAL, self=True)
    distance_EE, _ = ff(TEST, TEST, self=True)
    distance_SS, _ = ff(FAKE, FAKE, self=True)

    distance_TS, distance_ST = ff(REAL, FAKE)
    distance_ES, distance_SE = ff(TEST, FAKE)
    distance_TE, distance_ET = ff(REAL, TEST)

    distance_TS.shape, distance_ST.shape, distance_TT.shape, distance_SS.shape
    distance_EE.shape, distance_SE.shape, distance_ES.shape

    aa_train = (
        np.sum(distance_TS > distance_TT) / distance_TT.shape[0]
        + np.sum(distance_ST > distance_SS) / distance_SS.shape[0]
    ) / 2

    aa_test = (
        np.sum(distance_ES > distance_EE) / distance_EE.shape[0]
        + np.sum(distance_SE > distance_SS) / distance_SS.shape[0]
    ) / 2

    aa_train_test = (
        np.sum(distance_TE > distance_TT) / distance_TT.shape[0]
        + np.sum(distance_ET > distance_EE) / distance_EE.shape[0]
    ) / 2

    metrics_sym = {
        "Adv ACC/AA_train_syn": aa_train,
        "Adv ACC/AA_test_syn": aa_test,
        "Adv ACC/AA_train_test": aa_train_test,
        "Adv ACC/NNAA": (aa_train - aa_test),
    }

    metrics_asym = {
        "Train-Fake": np.sum(distance_TS > distance_TT) / distance_TT.shape[0]
        - distance_TT.shape[0] / (distance_TT.shape[0] + distance_SS.shape[0]),
        "Fake-Train": np.sum(distance_ST > distance_SS) / distance_SS.shape[0]
        - distance_SS.shape[0] / (distance_TT.shape[0] + distance_SS.shape[0]),
        "Test-Fake": np.sum(distance_ES > distance_EE) / distance_EE.shape[0]
        - distance_EE.shape[0] / (distance_EE.shape[0] + distance_SS.shape[0]),
        "Fake-Test": np.sum(distance_SE > distance_SS) / distance_SS.shape[0]
        - distance_SS.shape[0] / (distance_EE.shape[0] + distance_SS.shape[0]),
        "Train-Test": np.sum(distance_TE > distance_TT) / distance_TT.shape[0]
        - distance_TT.shape[0] / (distance_TT.shape[0] + distance_EE.shape[0]),
        "Test-Train": np.sum(distance_ET > distance_EE) / distance_EE.shape[0]
        - distance_EE.shape[0] / (distance_TT.shape[0] + distance_EE.shape[0]),
    }

    metrics_asym_bl = {
        # "Train-Fake-bl": distance_TT.shape[0]/(distance_TT.shape[0]+distance_SS.shape[0]),
        # "Fake-Train-bl": distance_SS.shape[0]/(distance_TT.shape[0]+distance_SS.shape[0]),
        # "Test-Fake-bl": distance_EE.shape[0]/(distance_EE.shape[0]+distance_SS.shape[0]),
        # "Fake-Test-bl": distance_SS.shape[0]/(distance_EE.shape[0]+distance_SS.shape[0]),
        # "Train-Test-bl": distance_TT.shape[0]/(distance_TT.shape[0]+distance_EE.shape[0]),
        # "Test-Train-bl": distance_EE.shape[0]/(distance_TT.shape[0]+distance_EE.shape[0]),
    }
    return metrics_sym#, metrics_asym, metrics_asym_bl




def compute_mia_knn(REAL, FAKE, TEST):
    # For MIA-kNN
    from sklearn.neighbors import KNeighborsClassifier
    from scipy.stats import gaussian_kde
    from scipy.stats import entropy, wasserstein_distance
    from sklearn.metrics import roc_auc_score

    knn = KNeighborsClassifier(n_neighbors=1)

    X = np.concatenate(
        [
            # REAL,
            FAKE
        ],
        axis=0,
    )  # [2*bs, hidden_dim]
    y = np.concatenate(
        [
            # np.ones(REAL.shape[0]),
            np.zeros(FAKE.shape[0])
        ],
        axis=0,
    )  # [2*bs, hidden_dim]

    knn.fit(X, y)

    test_nearest_dist, test_nearest_ids = knn.kneighbors(TEST, return_distance=True)
    train_nearest_dist, train_nearest_ids = knn.kneighbors(REAL, return_distance=True)

    if test_nearest_dist.shape[1] > 1:  # if more than 1 neighbor
        test_nearest_dist = test_nearest_dist.mean(1)
        train_nearest_dist = train_nearest_dist.mean(1)

    # fit non-parametric density
    kde_train = gaussian_kde(train_nearest_dist.flatten())
    kde_test = gaussian_kde(test_nearest_dist.flatten())

    def jensen_shannon_divergence(p, q):
        # Calculate the average distribution
        m = 0.5 * (p + q)

        # Calculate the Jensen-Shannon Divergence
        jsd = 0.5 * (entropy(p, m) + entropy(q, m))
        return jsd

    max_dist = max(train_nearest_dist.max(), test_nearest_dist.max())
    x_values = np.linspace(0, max_dist, 100)
    pdf_train = kde_train(x_values)
    pdf_test = kde_test(x_values)

    wasserstein_distance(pdf_train, pdf_test)
    jensen_shannon_divergence(pdf_train, pdf_test)

    metrics = {
        "MIA/WD": wasserstein_distance(pdf_train, pdf_test),
        "MIA/JSD": jensen_shannon_divergence(pdf_train, pdf_test),
        "MIA/knn-auroc": roc_auc_score(
            np.concatenate(
                [np.ones_like(train_nearest_dist), np.zeros_like(test_nearest_dist)],
                axis=0,
            ),
            np.concatenate([train_nearest_dist, test_nearest_dist], axis=0),
        ),
    }

    # # plot PDFs
    # fig = go.Figure()
    # _ = fig.add_trace(go.Scatter(x=x_values, y=kde_train(x_values),name="Train"))
    # _ = fig.add_trace(go.Scatter(x=x_values, y=kde_test(x_values),name="Test"))
    # fig.show()

    # # plot normalized histograms
    # fig = go.Figure()
    # _ = fig.add_trace(go.Histogram(x=train_nearest_dist.flatten(), histnorm='probability density',name="Train"))
    # _ = fig.add_trace(go.Histogram(x=test_nearest_dist.flatten(), histnorm='probability density',name="Test"))
    # fig.show()

    return metrics



def evaluate(inputs, wandb_task_name="DEBUG", config={}):

    # folderName = f"{opt.method}-r{opt.ratio}-{opt.cwgan_path}-{opt.pixgan_path}"

    # Create wandb run to save the results

    state_vars = inputs["state_vars"]
    col_labels = inputs["col_labels"]
    print(col_labels)
    CORR_METHOD = "ffill"
    CORR_TH = 0.2
    MODEL_NAME = "TimEHR"
    DATASET = "p12"

    # if not os.path.exists(f"./Results/{wandb_task_name}/"):
    #     os.makedirs(f"./Results/{wandb_task_name}/")
    wandb.init(
        config=config,
        project="TimEHR-Eval",
        name=wandb_task_name,
        reinit=True,
        # dir=f"./Results/{wandb_task_name}/TimEHR-Eval",
    )

    df_ts_fake, df_static_fake = inputs["df_ts_fake"], inputs["df_static_fake"]
    df_ts_train, df_static_train = inputs["df_ts_train"], inputs["df_static_train"]
    df_ts_test, df_static_test = inputs["df_ts_test"], inputs["df_static_test"]



    # compataiblity with the previous version
    df_train_fake, df_static_fake = df_ts_fake, df_static_fake
    df_train_real, df_static_real = df_ts_train, df_static_train
    df_test, df_static = df_ts_test, df_static_test







    # compute temporal correlation
    if DATASET in ["p12", "p19", "mimic-big"]:
        CORR_METHOD = "ffill"
    else:
        CORR_METHOD = "ffill"
    metric = plot_corr(
        df_train_real,
        df_train_fake,
        df_test,
        state_vars,
        corr_method=CORR_METHOD,
        corr_th=CORR_TH,
    )


    
    X_train, y_train = xgboost_embeddings(df_ts_train,df_static_train, state_vars,col_labels)
    X_fake, y_fake = xgboost_embeddings(df_ts_fake,df_static_fake, state_vars,col_labels)

    X_test_by_train, y_test = xgboost_embeddings(
        df_ts_test, df_static_test ,state_vars,col_labels, 
    ) # df_base=df_ts_train
    # X_test_by_fakke, _ = xgboost_embeddings(df_ts_test,df_static_test, state_vars,col_labels, )# df_base=df_ts_fake


    # # compataiblity with the previous version
    # df_train_fake, df_static_fake = df_ts_fake, df_static_fake
    # df_train_real, df_static_real = df_ts_train, df_static_train
    # df_test, df_static = df_ts_test, df_static_test

    X_train_real, y_train_real = X_train, y_train
    X_train_fake, y_train_fake = X_fake, y_fake

    X_test_real, y_test_real = X_test_by_train, y_test
    # X_test_fake, y_test_fake = X_test_by_fake, y_test

    # df_train_real = df_train_real.merge(df_static_real[['RecordID','Label']],on=['RecordID'],how='inner')
    # df_train_fake = df_train_fake.merge(df_static_fake[['RecordID','Label']],on=['RecordID'],how='inner')
    # df_test = df_test.merge(df_static[['RecordID','Label']],on=['RecordID'],how='inner')

    # X_train_real, X_train_fake, X_test_real, X_test_fake,\
    # y_train_real, y_train_fake, y_test_real, y_test_fake = xgboost_embeddings(df_train_real, df_train_fake, df_test, state_vars)

    # if opt.privacy_emb=='summary':
    #     LL = 4*len(state_vars)
    #     X_real = pd.concat([X_train_real.fillna(0).iloc[:,:LL], y_train_real], axis=1)
    #     X_fake = pd.concat([X_train_fake.fillna(0).iloc[:,:LL], y_train_fake], axis=1)
    #     X_test = pd.concat([X_test_real.fillna(0).iloc[:,:LL], y_test_real], axis=1)
    # elif opt.privacy_emb=='summary2':
    
    
    # save_examples(
    #     torch.from_numpy(inputs["train_data"][:10]),
    #     torch.from_numpy(inputs["fake_data"][:10]),
    #     n_ts=len(state_vars),
    #     epoch_no=-1,
    # )

    # compute utility
    if DATASET in ["p12", "p19", "mimic"]:
        # metrics = compute_utility(
        #     X_train_real,
        #     X_train_fake,
        #     X_test_real,
        #     X_test_fake,
        #     y_train_real,
        #     y_train_fake,
        #     y_test_real,
        #     y_test_fake,
        # )
        # wandb.log(metrics)

        train_fake_ratio=[
            (0,1),
            
            (0.1,1), (0.1,0),
            (0.2,1), (0.2,0),
            
            (0.5,1), (0.5,0),

            (1,1),
            
            (1,0)
            ]

        metrics = compute_utility2(X_train, y_train, X_test_by_train, y_test, X_fake, y_fake,train_fake_ratio=train_fake_ratio)

    
    LL = 4 * len(state_vars)
    X_real = pd.concat([X_train_real.fillna(0).iloc[:, :LL], y_train_real], axis=1)
    X_fake = pd.concat([X_train_fake.fillna(0).iloc[:, :LL], y_train_fake], axis=1)
    X_test = pd.concat([X_test_real.fillna(0).iloc[:, :LL], y_test_real], axis=1)




    # now resample to 10k
    # N_samples = min(1000000, len(X_test))
    N_samples = min(len(X_fake), len(X_real), len(X_test))
    # N_samples=5000
    print(f"\n[info] resampling to {N_samples}")
    X_real = X_real.sample(N_samples, random_state=42, replace=False)
    X_fake = X_fake.sample(N_samples, random_state=42, replace=False)
    X_test = X_test.sample(N_samples, random_state=42, replace=False)

    # IMPORTANT: we have some variables that are sort of discrete. Hence the histogram of real and test is no longer similar to a continuous variable. Synthcity will somehow detects this fact and will report low scores.
    X_real = X_real + np.random.normal(0, 0.00001, X_real.shape)
    X_fake = X_fake + np.random.normal(0, 0.00001, X_fake.shape)
    X_test = X_test + np.random.normal(0, 0.00001, X_test.shape)

    # SYNTHCITY
    # fill nan or inf values with 0

    # print number of nan and inf values
    print('[info] number of nan and inf values')
    print(X_fake.isnull().sum().sum(), X_fake.isin([np.inf, -np.inf]).sum().sum())
    print(X_real.isnull().sum().sum(), X_real.isin([np.inf, -np.inf]).sum().sum())
    print(X_test.isnull().sum().sum(), X_test.isin([np.inf, -np.inf]).sum().sum())
    X_fake =  X_fake.replace([np.inf, -np.inf], np.nan)
    X_fake = X_fake.fillna(X_fake.mean())
    X_real =  X_real.replace([np.inf, -np.inf], np.nan)
    X_real = X_real.fillna(X_real.mean())
    X_test =  X_test.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.fillna(X_test.mean())


    metrics = compute_synthcity(X_real, X_fake, X_test)
    wandb.log(metrics)

    print(X_real.columns)
    
    REAL = X_train_real.fillna(0).iloc[:, :LL].values
    FAKE = X_train_fake.fillna(0).iloc[:, :LL].values
    TEST = X_test_real.fillna(0).iloc[:, :LL].values

    print('[info] number of nan and inf values')
    print(np.isnan(REAL).sum().sum(), np.isinf(REAL).sum().sum())
    print(np.isnan(FAKE).sum().sum(), np.isinf(FAKE).sum().sum())
    print(np.isnan(TEST).sum().sum(), np.isinf(TEST).sum().sum())
    
    # compute privacy NNAA
    metrics_sym, metrics_asym, metrics_asym_bl = compute_nnaa(REAL, FAKE, TEST)
    wandb.log(metrics_sym)
    # wandb.log(metrics_asym)
    # wandb.log(metrics_asym_bl)

    # compute privacy MIA-kNN
    metrics = compute_mia_knn(REAL, FAKE, TEST)
    wandb.log(metrics)

    # plot tsne
    fig_tsne = plot_tsne(REAL, FAKE, N=10000)
    wandb.log({"t-SNE": wandb.Plotly(fig_tsne)})

    wandb.finish()


