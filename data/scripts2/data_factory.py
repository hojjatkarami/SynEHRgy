import random
from collections import UserDict
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
import pandas as pd
import torch
from datasets import Dataset as HFDataset

from torch.utils.data import IterableDataset, get_worker_info


def prepare_vocab_and_mappings(
    vocab_path: str | Path,
    min_count: int = 1000,
    max_vocabs: int | None = None,
    add_disc_tokens: bool = False,
):
    vocab = pd.read_parquet(vocab_path)
    vocab = vocab[vocab["group_count_by_id"] > min_count].sort_values(
        "group_count_by_id", ascending=False
    )

    vocab_list = [str(v) for v in set(vocab["concept_id"].tolist())]
    if max_vocabs is not None:
        mask = (~vocab["concept_id"].duplicated()).cumsum() <= max_vocabs
        vocab = vocab[mask]
        logger.info(f"limited vocab to {max_vocabs} entries.")

    # clip max_value to 2*99 percentile
    vocab.loc[vocab["max_value"] > 2 * vocab["p99"], "max_value"] = 2 * vocab["p99"]
    logger.info("clipped max_value to 2*99 percentile")

    vocab_list = vocab["concept_id"].unique().tolist()

    print(len(vocab_list))
    print(vocab.shape, vocab["concept_id"].nunique())

    dict_map = (
        vocab[["concept_uid", "concept_id"]]
        .set_index("concept_uid")["concept_id"]
        .to_dict()
    )

    bin_edges_uid = vocab.set_index("concept_uid")[
        [
            "min_value",
            "p01",
            "p025",
            "p05",
            "p10",
            "p25",
            "p50",
            "p75",
            "p95",
            "p975",
            "p99",
            "max_value",
        ]
    ].to_dict(orient="index")
    bin_edges_uid = {key: list(value.values()) for key, value in bin_edges_uid.items()}
    # ensure monotonic
    for values in bin_edges_uid.values():
        for i in range(len(values) - 1):
            if values[i] >= values[i + 1]:
                values[i + 1] = values[i] + 1e-6

    bin_edges_cid = (
        vocab.groupby("concept_id")
        .head(1)
        .set_index("concept_id")[
            [
                "min_value",
                "p01",
                "p025",
                "p05",
                "p10",
                "p25",
                "p50",
                "p75",
                "p95",
                "p975",
                "p99",
                "max_value",
            ]
        ]
        .to_dict(orient="index")
    )

    bin_edges_cid = {key: list(value.values()) for key, value in bin_edges_cid.items()}
    # print(vocab[vocab['concept_id'].str.contains('4148615_thousand per microliter')])
    # print(bin_edges_cid['4148615_thousand per microliter'])
    # term
    # ensure monotonic
    for values in bin_edges_cid.values():
        for i in range(len(values) - 1):
            if values[i] >= values[i + 1]:
                values[i + 1] = values[i] + 1e-6

    bin_labels = [
        "min_p01",
        "p01_p025",
        "p025_p05",
        "p05_p10",
        "p10_p25",
        "p25_p50",
        "p50_p75",
        "p75_p95",
        "p95_p975",
        "p975_p99",
        "p99_max",
    ]
    bin_label_weights = [
        0.01,
        0.015,
        0.025,
        0.05,
        0.1,
        0.25,
        0.25,
        0.2,
        0.025,
        0.015,
        0.01,
    ]

    disc_dt_edges = [0, 1.1, 7.1, 14.1, 30.1, 90.1, 180.1, 365.1, np.inf]
    disc_dt_labels = [
        "<d1>",
        "<d7>",
        "<d14>",
        "<d30>",
        "<d90>",
        "<d180>",
        "<d365>",
        "<d>365>",
    ]

    if add_disc_tokens:
        vocab_list += bin_labels
        vocab_list += disc_dt_labels

    header_tokens = [
        "<s>",
        "</s>",
        "<v>",
        "</v>",
    ]
    vocab_list += header_tokens

    output = {
        "dict_map": dict_map,
        "bin_edges_uid": bin_edges_uid,
        "bin_edges_cid": bin_edges_cid,
        "bin_labels": bin_labels,
        "disc_dt_edges": disc_dt_edges,
        "disc_dt_labels": disc_dt_labels,
        "vocab_list": vocab_list,
        "df_vocab": vocab,
        "bin_label_weights": bin_label_weights,
    }

    return output

