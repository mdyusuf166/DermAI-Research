import argparse
import logging
from pathlib import Path

from src.train import train_and_benchmark

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def main(config_path: str) -> None:
    train_and_benchmark(config_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the HAM10000 model benchmarking training pipeline.')
    parser.add_argument('--config', default='configs/default.yaml', help='Path to experiment configuration YAML.')
    args = parser.parse_args()
    main(args.config)
