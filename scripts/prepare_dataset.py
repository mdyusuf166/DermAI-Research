import argparse
import logging
from pathlib import Path

from src.data_pipeline import build_dataset
from src.utils import load_config, ensure_dirs

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def main(config_path: str) -> None:
    config = load_config(config_path)
    dataset_path = config['dataset_path']
    results_dir = config.get('results_dir', 'results')
    ensure_dirs(results_dir)

    logger.info('Loading metadata and preparing dataset splits from %s', dataset_path)
    train_df, val_df, test_df, group_col = build_dataset(config, dataset_path, results_dir)

    logger.info('Dataset audit saved to %s', Path(results_dir) / 'dataset_audit.csv')
    logger.info('Split file saved to %s', Path(results_dir) / 'data_split.csv')
    logger.info('Class distribution saved to %s', Path(results_dir) / 'class_distribution.csv')
    logger.info('Class distribution plot saved to %s', Path(results_dir) / 'class_distribution.png')
    logger.info('Group-based split completed using column: %s', group_col)
    logger.info('Split sizes: train=%d, val=%d, test=%d', len(train_df), len(val_df), len(test_df))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare and audit HAM10000 dataset splits.')
    parser.add_argument('--config', default='configs/default.yaml', help='Path to experiment configuration YAML.')
    args = parser.parse_args()
    main(args.config)
