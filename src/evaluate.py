import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            total_loss += criterion(outputs, targets).item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    report = classification_report(all_targets, all_preds, output_dict=True, zero_division=0)
    metrics = {
        'accuracy': accuracy_score(all_targets, all_preds),
        'balanced_accuracy': balanced_accuracy_score(all_targets, all_preds),
        'macro_precision': precision_score(all_targets, all_preds, average='macro', zero_division=0),
        'macro_recall': recall_score(all_targets, all_preds, average='macro', zero_division=0),
        'macro_f1': f1_score(all_targets, all_preds, average='macro', zero_division=0),
        'weighted_precision': precision_score(all_targets, all_preds, average='weighted', zero_division=0),
        'weighted_recall': recall_score(all_targets, all_preds, average='weighted', zero_division=0),
        'weighted_f1': f1_score(all_targets, all_preds, average='weighted', zero_division=0),
        'loss': total_loss / max(1, len(all_targets)),
        'classification_report': report,
        'confusion_matrix': confusion_matrix(all_targets, all_preds).tolist(),
    }
    return metrics


def save_json(data: Dict, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as handle:
        json.dump(data, handle, indent=2)


def save_comparison_table(df: pd.DataFrame, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)


def evaluate_history(history: List[Dict], output_path: Path) -> None:
    if not history:
        return
    epochs = [entry['epoch'] for entry in history]
    train_loss = [entry['train_loss'] for entry in history]
    val_loss = [entry['val_loss'] for entry in history]
    val_metric = [entry['val_macro_f1'] for entry in history]

    plt.figure(figsize=(10, 4))
    plt.plot(epochs, train_loss, label='train_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(epochs, val_metric, label='val_macro_f1')
    plt.xlabel('epoch')
    plt.ylabel('macro_f1')
    plt.title('Validation macro-F1 over epochs')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path.with_name(f'{output_path.stem}_metric.png'))
    plt.close()


def plot_model_comparison(df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='model_name', y='test_macro_f1')
    plt.title('Model Comparison by Test macro-F1')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_confusion_matrix(matrix: List[List[int]], class_names: List[str], output_path: Path) -> None:
    cm = np.array(matrix)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_per_class_f1(report: Dict[str, Dict[str, float]], class_names: List[str], output_path: Path) -> None:
    labels = [name for name in class_names if name in report]
    f1_scores = [report[label]['f1-score'] for label in labels]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=labels, y=f1_scores)
    plt.title('Per-class F1 scores')
    plt.xlabel('class')
    plt.ylabel('F1 score')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
