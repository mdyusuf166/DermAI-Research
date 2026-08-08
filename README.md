# HAM10000 Skin Cancer Detection Benchmark

This repository contains a reproducible Phase 2 / Phase 3 pipeline for the HAM10000 skin lesion dataset.
It supports dataset audit and grouped train/validation/test splits, then trains and benchmarks multiple models with leakage-safe evaluation.

## Project structure

- `configs/default.yaml` - default experiment configuration
- `scripts/prepare_dataset.py` - run Phase 2 dataset audit and grouped split generation
- `scripts/train.py` - train and benchmark the model suite
- `scripts/evaluate.py` - generate aggregated evaluation reports and plots
- `src/data_pipeline.py` - metadata loading, path resolution, grouped split creation, dataset builder
- `src/models/build_models.py` - model factory for baseline CNN and TIMM backbones
- `src/train.py` - benchmark training loop, checkpointing, class imbalance strategy
- `src/evaluate.py` - metrics computation, result saving, plot generation
- `src/utils.py` - reproducibility helpers, config loading, common utilities
- `checkpoints/` - saved best model weights
- `results/` - dataset audit artifacts, split files, comparison tables, and figures

## Requirements

The project uses PyTorch, TorchVision, TIMM, Albumentations, OpenCV, and standard ML libraries.
Install dependencies with:

```bash
python -m pip install -r requirements.txt
```

> Note: The local Python environment must support the installed `torch` binary. If you see `WinError 193` or DLL import failures, install a matching CUDA/cpu build for your OS and Python version.

## Dataset setup

Place the HAM10000 dataset inside:

```text
SUAPUai/skin-cancer-detection-ham10000/
```

and ensure `configs/default.yaml` points to the dataset root:

```yaml
dataset_path: "./SUAPUai/skin-cancer-detection-ham10000/"
metadata_filename: "HAM10000_metadata.csv"
```

## Phase 2: Prepare dataset splits

Run the dataset preparation pipeline to create audit files, class distribution reports, and a leakage-safe grouped split.

```bash
python scripts/prepare_dataset.py --config configs/default.yaml
```

Outputs:

- `results/dataset_audit.csv`
- `results/data_split.csv`
- `results/class_distribution.csv`
- `results/class_distribution.png`

## Phase 3: Train and benchmark models

Train the model suite using the reproducible benchmark pipeline:

```bash
python scripts/train.py --config configs/default.yaml
```

This script will:

- build grouped train/val/test datasets from `results/data_split.csv`
- train each model in `model_names`
- save the best checkpoint by validation `macro_f1`
- evaluate the best model on the test set
- save `results/model_comparison.csv` and `results/model_comparison.json`
- save training history plots to `results/figures/`

## Evaluation

After training, generate evaluation figures and a summary report with:

```bash
python scripts/evaluate.py --config configs/default.yaml
```

This will create:

- `results/model_comparison.csv`
- `results/model_comparison.json`
- `results/evaluation_summary.json`
- `results/figures/model_comparison.png`
- `results/figures/<best_model>_confusion_matrix.png`
- `results/figures/<best_model>_per_class_f1.png`

## Configuration

Key parameters in `configs/default.yaml`:

- `image_size` - input resolution for all models
- `batch_size` - training batch size
- `epochs` - number of training epochs
- `learning_rate` - optimizer learning rate
- `weight_decay` - weight decay for AdamW
- `scheduler` - learning rate scheduler type
- `class_strategy` - imbalance handling strategy: `standard`, `class_weighted_loss`, `focal_loss`, or `weighted_sampling`
- `model_names` - list of models to benchmark

## Notes

- The current benchmark includes:
  - `baseline_cnn`
  - `resnet18`
  - `resnet34`
  - `resnet50`
  - `densenet121`
  - `efficientnet_b0`
- The split is leakage-aware and based on `patient_id` or `lesion_id` from the metadata.
- Use the generated `results/data_split.csv` as the canonical split for all experiments.

## Troubleshooting

- If training fails because `torch` cannot import, verify the installed PyTorch wheel matches your Python version and Windows architecture.
- If dataset images are missing, `src/data_pipeline.py` will drop rows with missing paths and warn you.
- If `data_split.csv` is absent, `scripts/train.py` will regenerate it using the Phase 2 pipeline.
