import os
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> Dict:
    import yaml

    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def validate_metadata(df: pd.DataFrame, metadata_path: str) -> pd.DataFrame:
    required_columns = ['image_id', 'dx']
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required metadata columns: {missing}")

    df = df.copy()
    df['image_id'] = df['image_id'].astype(str)
    df['dx'] = df['dx'].astype(str)

    if 'lesion_id' in df.columns:
        df['lesion_id'] = df['lesion_id'].astype(str)
    if 'patient_id' in df.columns:
        df['patient_id'] = df['patient_id'].astype(str)

    duplicates = df['image_id'].duplicated().sum()
    if duplicates > 0:
        raise ValueError(f"Found {duplicates} duplicate image_id values in {metadata_path}")

    return df


def get_group_column(df: pd.DataFrame) -> Optional[str]:
    if 'patient_id' in df.columns and df['patient_id'].notna().any():
        return 'patient_id'
    if 'lesion_id' in df.columns and df['lesion_id'].notna().any():
        return 'lesion_id'
    return None


def perform_grouped_split(
    df: pd.DataFrame,
    group_col: Optional[str],
    test_size: float,
    validation_size: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    if group_col is None:
        raise ValueError('No grouping column available for leakage-safe split.')

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    groups = df[group_col]
    train_val_idx, test_idx = next(splitter.split(df, groups=groups))
    train_val = df.iloc[train_val_idx].reset_index(drop=True)
    test = df.iloc[test_idx].reset_index(drop=True)

    val_size_relative = validation_size / (1.0 - test_size)
    splitter_val = GroupShuffleSplit(n_splits=1, test_size=val_size_relative, random_state=seed)
    train_idx, val_idx = next(splitter_val.split(train_val, groups=train_val[group_col]))
    train = train_val.iloc[train_idx].reset_index(drop=True)
    val = train_val.iloc[val_idx].reset_index(drop=True)

    return train, val, test


def save_audit_files(
    df: pd.DataFrame,
    audit_path: str,
    split_df: pd.DataFrame,
    split_path: str,
    class_path: str,
    distribution_path: str,
) -> None:
    df.to_csv(audit_path, index=False)
    split_df.to_csv(split_path, index=False)
    distribution = (
        df['dx']
        .value_counts()
        .rename_axis('dx')
        .reset_index(name='count')
    )
    distribution['percentage'] = distribution['count'] / distribution['count'].sum() * 100.0
    distribution.to_csv(class_path, index=False)

    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(10, 6))
    sns.barplot(data=distribution, x='dx', y='count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(distribution_path)
    plt.close()


def check_group_overlap(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, group_col: Optional[str]) -> Dict[str, int]:
    overlaps = {}
    if group_col is None:
        return overlaps

    train_groups = set(train[group_col])
    val_groups = set(val[group_col])
    test_groups = set(test[group_col])

    overlaps['train_val'] = len(train_groups.intersection(val_groups))
    overlaps['train_test'] = len(train_groups.intersection(test_groups))
    overlaps['val_test'] = len(val_groups.intersection(test_groups))
    return overlaps
