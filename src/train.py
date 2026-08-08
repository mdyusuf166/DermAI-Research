import argparse
import argparse
import json
import logging
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data_pipeline import HAM10000Dataset, build_dataset, get_transforms
from src.evaluation import compute_metrics, evaluate_history, save_json, save_comparison_table
from src.models import build_model, freeze_backbone, unfreeze_backbone, is_transfer_learning_model
from src.utils import load_config, set_seed, ensure_dirs, get_device

logger = logging.getLogger(__name__)


def _get_class_weights(labels: pd.Series) -> torch.Tensor:
    counts = labels.value_counts().sort_index().astype(float)
    weights = counts.sum() / counts
    return torch.tensor(weights.values, dtype=torch.float32)


def _get_sampler(labels: pd.Series) -> Optional[WeightedRandomSampler]:
    class_counts = labels.value_counts().sort_index()
    if len(class_counts) <= 1:
        return None

    sample_weights = labels.map(lambda x: class_counts.sum() / class_counts.loc[x]).values
    sample_weights = torch.tensor(sample_weights, dtype=torch.float32)
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weight = self.weight.to(inputs.device) if self.weight is not None else None
        ce = nn.CrossEntropyLoss(weight=weight, reduction='none')
        logpt = -ce(inputs, targets)
        pt = torch.exp(logpt)
        loss = -(1 - pt) ** self.gamma * logpt
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss


def build_criterion(config: Dict, class_weights: Optional[torch.Tensor]) -> nn.Module:
    strategy = config.get('class_strategy', 'standard')
    if strategy == 'class_weighted_loss':
        return nn.CrossEntropyLoss(weight=class_weights.to(get_device()))
    if strategy == 'focal_loss':
        return FocalLoss(gamma=config.get('focal_gamma', 2.0), weight=class_weights)
    return nn.CrossEntropyLoss()


def build_dataloaders(
    config: Dict,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_col: str = 'label_idx',
) -> Tuple[DataLoader, DataLoader, DataLoader, Optional[torch.Tensor], List[str]]:
    image_size = config['image_size']
    train_transform = get_transforms(image_size, phase='train')
    eval_transform = get_transforms(image_size, phase='val')

    class_names = train_df[['label_idx', 'dx']].drop_duplicates().sort_values('label_idx')['dx'].tolist()
    dataset_path = config['dataset_path']

    train_dataset = HAM10000Dataset(train_df, dataset_path, transform=train_transform, label_col=label_col)
    val_dataset = HAM10000Dataset(val_df, dataset_path, transform=eval_transform, label_col=label_col)
    test_dataset = HAM10000Dataset(test_df, dataset_path, transform=eval_transform, label_col=label_col)

    class_weights = _get_class_weights(train_df[label_col])
    sampler = None
    if config.get('class_strategy') == 'weighted_sampling':
        sampler = _get_sampler(train_df[label_col])

    batch_size = config['batch_size']
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=config.get('num_workers', 2),
        pin_memory=True,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=config.get('num_workers', 2), pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=config.get('num_workers', 2), pin_memory=True)

    return train_loader, val_loader, test_loader, class_weights, class_names


def _set_torch_determinism() -> None:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    config: Dict,
    checkpoint_path: str,
) -> Tuple[Dict, List[Dict]]:
    best_metrics = {
        'epoch': 0,
        'val_loss': float('inf'),
        'val_macro_f1': 0.0,
    }
    history: List[Dict] = []
    best_state = None

    for epoch in range(1, config['epochs'] + 1):
        model.train()
        epoch_loss = 0.0
        step = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)
            step += inputs.size(0)

        train_loss = epoch_loss / max(1, len(train_loader.dataset))
        val_metrics = compute_metrics(model, val_loader, device)
        val_loss = val_metrics['loss']
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_macro_f1': val_metrics['macro_f1'],
            'val_accuracy': val_metrics['accuracy'],
        })

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        if val_metrics['macro_f1'] > best_metrics['val_macro_f1']:
            best_metrics.update(
                epoch=epoch,
                val_loss=val_loss,
                val_macro_f1=val_metrics['macro_f1'],
                val_accuracy=val_metrics['accuracy'],
            )
            best_state = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'best_metrics': best_metrics,
            }
            torch.save(best_state, checkpoint_path)
            logger.info('Saved best checkpoint to %s at epoch %d', checkpoint_path, epoch)

    if best_state is None:
        raise RuntimeError('Training did not produce a checkpoint for the best model.')

    return best_metrics, history


def load_split_data(config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    results_dir = Path(config['results_dir'])
    split_path = results_dir / 'data_split.csv'
    if not split_path.exists():
        logger.info('Split file not found; generating dataset audit and split with Phase 2 pipeline.')
        train_df, val_df, test_df, group_col = build_dataset(config, config['dataset_path'], config['results_dir'])
        return train_df, val_df, test_df, group_col

    split_df = pd.read_csv(split_path)
    group_col = 'patient_id' if 'patient_id' in split_df.columns else 'lesion_id'
    if group_col not in split_df.columns:
        raise ValueError('No group column found in data split file.')

    return (
        split_df[split_df['partition'] == 'train'].reset_index(drop=True),
        split_df[split_df['partition'] == 'val'].reset_index(drop=True),
        split_df[split_df['partition'] == 'test'].reset_index(drop=True),
        group_col,
    )


def train_and_benchmark(config_path: str) -> None:
    config = load_config(config_path)
    set_seed(config['seed'])
    _set_torch_determinism()
    device = get_device()

    results_dir = Path(config['results_dir'])
    checkpoints_dir = Path(config.get('checkpoints_dir', 'checkpoints'))
    figures_dir = Path(config.get('figures_dir', results_dir / 'figures'))
    ensure_dirs(results_dir, checkpoints_dir, figures_dir)

    train_df, val_df, test_df, group_col = load_split_data(config)
    train_loader, val_loader, test_loader, class_weights, class_names = build_dataloaders(
        config, train_df, val_df, test_df
    )

    results = []
    for model_name in config['model_names']:
        logger.info('Training model: %s', model_name)
        model = build_model(model_name, num_classes=len(class_names), pretrained=config.get('pretrained', True))

        if config.get('freeze_backbone_epochs', 0) > 0 and is_transfer_learning_model(model_name):
            logger.info('Freezing backbone for first %d epoch(s) of %s', config['freeze_backbone_epochs'], model_name)
            freeze_backbone(model)

        criterion = build_criterion(config, class_weights)
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2) if config.get('scheduler', 'ReduceLROnPlateau') == 'ReduceLROnPlateau' else torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        checkpoint_path = checkpoints_dir / f'{model_name}_best.pth'
        best_metrics, history = train_loop(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            scheduler,
            device,
            config,
            str(checkpoint_path),
        )

        logger.info('Best validation macro F1 for %s: %.4f', model_name, best_metrics['val_macro_f1'])

        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        test_metrics = compute_metrics(model, test_loader, device)

        model_results = {
            'model_name': model_name,
            'checkpoint': str(checkpoint_path),
            'config': config,
            'group_column': group_col,
            'best_validation': best_metrics,
            'test_metrics': test_metrics,
            'class_names': class_names,
        }
        results.append(model_results)

        evaluate_history(history, figures_dir / f'{model_name}_history.png')

    summary_path = results_dir / 'model_comparison.json'
    save_json({'models': results}, str(summary_path))

    rows = []
    for result in results:
        metrics = result['test_metrics']
        rows.append({
            'model_name': result['model_name'],
            'test_accuracy': metrics['accuracy'],
            'test_balanced_accuracy': metrics['balanced_accuracy'],
            'test_macro_precision': metrics['macro_precision'],
            'test_macro_recall': metrics['macro_recall'],
            'test_macro_f1': metrics['macro_f1'],
            'test_weighted_f1': metrics['weighted_f1'],
        })
    comparison_df = pd.DataFrame(rows)
    save_comparison_table(comparison_df, str(results_dir / 'model_comparison.csv'))

    logger.info('Saved model comparison files: %s, %s', results_dir / 'model_comparison.csv', summary_path)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    parser = argparse.ArgumentParser(description='Train and benchmark HAM10000 models.')
    parser.add_argument('--config', default='configs/default.yaml', help='Path to configuration YAML.')
    args = parser.parse_args()
    train_and_benchmark(args.config)
