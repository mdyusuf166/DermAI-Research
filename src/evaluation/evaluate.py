import json
import os
from typing import Dict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_model(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())
    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    metrics = {
        'accuracy': accuracy_score(all_targets, all_preds),
        'balanced_accuracy': balanced_accuracy_score(all_targets, all_preds),
        'macro_precision': precision_score(all_targets, all_preds, average='macro', zero_division=0),
        'macro_recall': recall_score(all_targets, all_preds, average='macro', zero_division=0),
        'macro_f1': f1_score(all_targets, all_preds, average='macro', zero_division=0),
        'weighted_precision': precision_score(all_targets, all_preds, average='weighted', zero_division=0),
        'weighted_recall': recall_score(all_targets, all_preds, average='weighted', zero_division=0),
        'weighted_f1': f1_score(all_targets, all_preds, average='weighted', zero_division=0),
        'classification_report': classification_report(all_targets, all_preds, output_dict=True, zero_division=0),
    }
    return metrics


def save_metrics(metrics: Dict, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)


def save_comparison_table(df: pd.DataFrame, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
