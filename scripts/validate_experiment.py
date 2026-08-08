import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from src.data_pipeline import find_metadata_file
from src.utils import load_config


def validate_metadata(df: pd.DataFrame) -> None:
    required = {'image_id', 'dx'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Missing metadata columns: {missing}')

    if df['image_id'].duplicated().any():
        raise ValueError('Duplicate image_id values found in metadata.')

    if df['dx'].isna().any():
        raise ValueError('Missing dx labels in metadata.')


def validate_class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    distribution = (
        df['dx']
        .value_counts()
        .rename_axis('dx')
        .reset_index(name='count')
    )
    distribution['percentage'] = distribution['count'] / distribution['count'].sum() * 100.0
    return distribution


def detect_group_column(df: pd.DataFrame) -> Optional[str]:
    if 'patient_id' in df.columns and df['patient_id'].notna().any():
        return 'patient_id'
    if 'lesion_id' in df.columns and df['lesion_id'].notna().any():
        return 'lesion_id'
    return None


def validate_split(df: pd.DataFrame, group_col: str, config: Optional[Dict] = None) -> None:
    partitions = set(df['partition'].unique())
    expected_partitions = {'train', 'val', 'test'}
    if partitions != expected_partitions:
        raise ValueError(f'Split partitions must contain exactly {expected_partitions}. Found: {partitions}')

    if group_col not in df.columns:
        raise ValueError(f'Missing grouping column {group_col} in split data.')

    groups = df.groupby('partition')[group_col].apply(set)
    if groups['train'].intersection(groups['val']):
        raise ValueError('Group leakage between train and val detected.')
    if groups['train'].intersection(groups['test']):
        raise ValueError('Group leakage between train and test detected.')
    if groups['val'].intersection(groups['test']):
        raise ValueError('Group leakage between val and test detected.')

    if config is not None:
        total = len(df)
        sizes = {partition: len(df[df['partition'] == partition]) for partition in expected_partitions}
        test_size = sizes['test'] / total
        train_val_size = sizes['train'] + sizes['val']
        val_relative = sizes['val'] / train_val_size if train_val_size else 0.0

        tolerance = 0.08
        if abs(test_size - config.get('test_size', 0.0)) > tolerance:
            raise ValueError(
                f'Test split fraction {test_size:.3f} differs from configured {config.get("test_size"):.3f} by more than {tolerance:.2f}'
            )
        if abs(val_relative - config.get('validation_size', 0.0) / (1.0 - config.get('test_size', 0.0))) > tolerance:
            raise ValueError(
                f'Validation split fraction relative to train+val {val_relative:.3f} differs from configured {config.get("validation_size") / (1.0 - config.get("test_size")):.3f} by more than {tolerance:.2f}'
            )


def validate_required_files(results_dir: Path) -> None:
    required_files = [
        'results/dataset_audit.csv',
        'results/data_split.csv',
        'results/class_distribution.csv',
    ]
    missing = [str(results_dir / Path(path).name) for path in required_files if not (Path(path).exists())]
    if missing:
        raise FileNotFoundError(f'Missing required result files: {missing}')


def main(config_path: str) -> int:
    config = load_config(config_path)
    results_dir = Path(config.get('results_dir', 'results'))
    dataset_dir = Path(config['dataset_path'])
    errors = []

    # Validate existing audit files
    audit_file = results_dir / 'dataset_audit.csv'
    split_file = results_dir / 'data_split.csv'
    try:
        if audit_file.exists():
            df = pd.read_csv(audit_file)
            validate_metadata(df)
            distribution = validate_class_distribution(df)
            print('Dataset audit loaded successfully:')
            print(distribution.to_string(index=False))
        else:
            raise FileNotFoundError(f'Missing audit file: {audit_file}')

        if split_file.exists():
            split_df = pd.read_csv(split_file)
            group_col = detect_group_column(split_df)
            if group_col is None:
                raise ValueError('No grouping column found in split file.')
            validate_split(split_df, group_col, config)
            print(f'Split file loaded successfully using group column: {group_col}')
            split_counts = split_df['partition'].value_counts().rename_axis('partition').reset_index(name='count')
            print(split_counts.to_string(index=False))
        else:
            raise FileNotFoundError(f'Missing split file: {split_file}')
    except Exception as exc:
        errors.append(f'Result validation error: {exc}')

    # Validate dataset metadata and image loading when dataset path is available
    if dataset_dir.exists():
        try:
            metadata_path = find_metadata_file(dataset_dir, config.get('metadata_filename'))
            raw_df = pd.read_csv(metadata_path)
            validate_metadata(raw_df)
            print(f'Dataset metadata loaded successfully from: {metadata_path}')
        except Exception as exc:
            errors.append(f'Dataset load error: {exc}')
    else:
        errors.append(f'Dataset path does not exist: {dataset_dir}')

    if errors:
        print('\n'.join(errors))
        return 1

    print('Experiment validation passed.')
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate HAM10000 experiment metadata and split integrity.')
    parser.add_argument('--config', default='configs/default.yaml', help='Experiment configuration file.')
    args = parser.parse_args()
    sys.exit(main(args.config))
