import json
import os
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from src.data.data_utils import set_seed


def train_epoch(model: nn.Module, loader: DataLoader, criterion, optimizer, device: torch.device):
    model.train()
    running_loss = 0.0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(loader.dataset)


def evaluate_epoch(model: nn.Module, loader: DataLoader, criterion, device: torch.device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            all_preds.append(outputs.softmax(dim=1).cpu())
            all_targets.append(targets.cpu())
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    pred_labels = all_preds.argmax(dim=1)
    macro_f1 = f1_score(all_targets.numpy(), pred_labels.numpy(), average='macro')
    return running_loss / len(loader.dataset), macro_f1


def save_checkpoint(state: Dict, checkpoint_path: str):
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(state, checkpoint_path)


def run_experiment(
    model,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion,
    optimizer,
    scheduler,
    device: torch.device,
    config: Dict,
    output_dir: str,
):
    set_seed(config['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    best_metrics = {
        'epoch': 0,
        'val_loss': float('inf'),
        'val_macro_f1': 0.0,
        'val_accuracy': 0.0,
    }
    history = []

    for epoch in range(1, config['epochs'] + 1):
        start = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_macro_f1 = evaluate_epoch(model, val_loader, criterion, device)
        epoch_time = time.time() - start
        state = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_macro_f1': val_macro_f1,
            'epoch_time': epoch_time,
        }
        history.append(state)

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        if val_macro_f1 > best_metrics['val_macro_f1']:
            best_metrics.update(
                epoch=epoch,
                val_loss=val_loss,
                val_macro_f1=val_macro_f1,
                val_accuracy=0.0,
            )
            checkpoint_path = os.path.join(output_dir, f'{model_name}_best.pth')
            save_checkpoint(
                {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'config': config,
                    'best_metrics': best_metrics,
                },
                checkpoint_path,
            )

    return best_metrics, history
