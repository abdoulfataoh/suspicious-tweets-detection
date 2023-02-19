# coding: utf-8

from src import settings
from src.data_loaders import DataLoader


__all__ = [
    'DataLoader',
]

settings.ASSETS_PATH.mkdir(parents=True, exist_ok=True)
settings.DATASET_PATH.mkdir(parents=True, exist_ok=True)
settings.METRICS_PATH.mkdir(parents=True, exist_ok=True)
