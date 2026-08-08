import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from src.evaluation import plot_confusion_matrix, plot_model_comparison, plot_per_class_f1, save_json
from src.utils import load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def main(config_path: str) -> None:
    config = load_config(config_path)
    results_dir = Path(config.get('results_dir', 'results'))
    figures_dir = Path(config.get('figures_dir', results_dir / 'figures'))
    results_path = results_dir / 'model_comparison.json'
    if not results_path.exists():
        raise FileNotFoundError('model_comparison.json not found. Run scripts/train.py first.')

    with open(results_path, 'r', encoding='utf-8') as handle:
        data = json.load(handle)

    models = data.get('models', [])
    if not models:
        raise ValueError('No model benchmark results found in model_comparison.json.')

    rows = []
    for model in models:
        metrics = model['test_metrics']
        rows.append({
            'model_name': model['model_name'],
            'test_macro_f1': metrics['macro_f1'],
            'test_accuracy': metrics['accuracy'],
            'test_balanced_accuracy': metrics['balanced_accuracy'],
            'test_weighted_f1': metrics['weighted_f1'],
        })

    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(results_dir / 'model_comparison.csv', index=False)
    plot_model_comparison(comparison_df, figures_dir / 'model_comparison.png')

    best_model = max(models, key=lambda item: item['test_metrics']['macro_f1'])
    class_names = best_model.get('class_names', [])
    confusion_path = figures_dir / f"{best_model['model_name']}_confusion_matrix.png"
    plot_confusion_matrix(best_model['test_metrics']['confusion_matrix'], class_names, confusion_path)
    plot_per_class_f1(best_model['test_metrics']['classification_report'], class_names, figures_dir / f"{best_model['model_name']}_per_class_f1.png")

    logger.info('Saved evaluation figures to %s', figures_dir)
    save_json({'summary': {'best_model': best_model['model_name']}}, str(results_dir / 'evaluation_summary.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate evaluation figures from benchmark results.')
    parser.add_argument('--config', default='configs/default.yaml', help='Path to experiment configuration YAML.')
    args = parser.parse_args()
    main(args.config)
