import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import pandas as pd
from albumentations.pytorch import ToTensorV2

from src.data.data_utils import validate_metadata, get_group_column, perform_grouped_split, save_audit_files, check_group_overlap
from src.utils import ensure_dirs, set_seed

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def find_metadata_file(dataset_path: Path, metadata_filename: Optional[str] = None) -> Path:
    if metadata_filename is not None:
        candidate = dataset_path / metadata_filename
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f'Metadata file not found: {candidate}')

    candidates = [
        dataset_path / 'HAM10000_metadata.csv',
        dataset_path / 'metadata.csv',
        dataset_path / 'HAM10000_metadata.tsv',
        dataset_path / 'HAM10000_metadata.txt',
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    for path in dataset_path.rglob('*HAM10000*metadata*.csv'):
        return path
    raise FileNotFoundError('Could not locate a metadata CSV file under dataset path.')


def load_metadata(dataset_path: str, metadata_filename: Optional[str] = None) -> pd.DataFrame:
    dataset_path = Path(dataset_path)
    metadata_path = find_metadata_file(dataset_path, metadata_filename)
    df = pd.read_csv(metadata_path)
    df = validate_metadata(df, str(metadata_path))
    df['dx'] = df['dx'].astype(str)
    df['image_id'] = df['image_id'].astype(str)
    return df


def build_image_index(dataset_path: str, image_folders: Optional[List[str]] = None) -> Dict[str, Path]:
    dataset_path = Path(dataset_path)
    image_index: Dict[str, Path] = {}
    search_roots = [dataset_path]
    if image_folders:
        search_roots.extend([dataset_path / folder for folder in image_folders])

    for root in search_roots:
        if not root.exists():
            continue
        for path in root.rglob('*'):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                image_index[path.stem] = path
    return image_index


def resolve_image_paths(df: pd.DataFrame, dataset_path: str, image_folders: Optional[List[str]] = None) -> pd.DataFrame:
    image_index = build_image_index(dataset_path, image_folders=image_folders)
    if not image_index:
        raise FileNotFoundError(f'No image files found beneath {dataset_path}')

    def _find_path(image_id: str) -> Optional[str]:
        if image_id in image_index:
            return str(image_index[image_id])
        lower = image_id.lower()
        if lower in image_index:
            return str(image_index[lower])
        for key, path in image_index.items():
            if key.lower() == lower:
                return str(path)
        return None

    df = df.copy()
    df['image_path'] = df['image_id'].apply(_find_path)
    missing = df['image_path'].isna()
    if missing.any():
        missing_count = missing.sum()
        logger.warning(
            'Missing image files for %d rows. These rows will be dropped from the dataset.',
            missing_count,
        )
        df = df.loc[~missing].reset_index(drop=True)
    return df


def add_label_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'label_idx' not in df.columns:
        df['label_idx'] = pd.factorize(df['dx'])[0]
    return df


def create_grouped_splits(
    df: pd.DataFrame,
    test_size: float,
    validation_size: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[str]]:
    group_col = get_group_column(df)
    if group_col is None:
        raise ValueError('No grouping column found for leakage-safe split. Add patient_id or lesion_id to metadata.')

    train, val, test = perform_grouped_split(
        df=df,
        group_col=group_col,
        test_size=test_size,
        validation_size=validation_size,
        seed=seed,
    )

    train = train.copy(); train['partition'] = 'train'
    val = val.copy(); val['partition'] = 'val'
    test = test.copy(); test['partition'] = 'test'
    return train, val, test, group_col


def build_split_dataframe(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([train, val, test], axis=0, ignore_index=True)


def save_pipeline_artifacts(
    df: pd.DataFrame,
    split_df: pd.DataFrame,
    results_dir: str,
    class_distribution_path: str,
    distribution_plot_path: str,
) -> None:
    ensure_dirs(results_dir)
    save_audit_files(
        df,
        audit_path=str(Path(results_dir) / 'dataset_audit.csv'),
        split_df=split_df,
        split_path=str(Path(results_dir) / 'data_split.csv'),
        class_path=str(Path(results_dir) / 'class_distribution.csv'),
        distribution_path=distribution_plot_path,
    )


def get_transforms(image_size: int, phase: str = 'train') -> A.Compose:
    if phase not in {'train', 'val', 'test'}:
        raise ValueError('phase must be one of train, val, or test')

    if phase == 'train':
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.Normalize(),
                ToTensorV2(),
            ]
        )

    return A.Compose([A.Resize(image_size, image_size), A.Normalize(), ToTensorV2()])


class HAM10000Dataset:
    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: str,
        transform: Optional[A.Compose] = None,
        image_id_col: str = 'image_id',
        label_col: str = 'label_idx',
    ):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.image_dir = Path(image_dir)
        self.image_id_col = image_id_col
        self.label_col = label_col

        if 'image_path' not in self.df.columns:
            self.df = resolve_image_paths(self.df, str(self.image_dir))

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        image_path = Path(row['image_path'])
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f'Unable to load image at {image_path}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32)

        if self.transform is not None:
            result = self.transform(image=image)
            image = result['image']

        label = int(row[self.label_col])
        return image, label


def build_dataset(
    config: Dict,
    dataset_path: str,
    results_dir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    set_seed(config.get('seed', 42))
    df = load_metadata(dataset_path, metadata_filename=config.get('metadata_filename'))
    df = add_label_index(df)
    df = resolve_image_paths(df, dataset_path, image_folders=config.get('image_folders'))
    train, val, test, group_col = create_grouped_splits(
        df=df,
        test_size=config['test_size'],
        validation_size=config['validation_size'],
        seed=config['seed'],
    )
    overlaps = check_group_overlap(train, val, test, group_col)
    if any(value > 0 for value in overlaps.values()):
        raise ValueError(f'Group leakage detected across partitions: {overlaps}')

    split_df = build_split_dataframe(train, val, test)
    save_pipeline_artifacts(
        df,
        split_df,
        results_dir,
        class_distribution_path=str(Path(results_dir) / 'class_distribution.csv'),
        distribution_plot_path=str(Path(results_dir) / 'class_distribution.png'),
    )
    return train, val, test, group_col
