# coding: utf-8

import logging

from src import settings
from src.data_loaders import DataLoader

logger = logging.getLogger(__name__)


__all__ = [
    'DataLoader',
]

# Create folder structure if not exist
logger.info('Create assets folder structure')
settings.ASSETS_FOLDER.mkdir(parents=True, exist_ok=True)
settings.DATASET_FOLDER.mkdir(parents=True, exist_ok=True)
settings.METRICS_FOLDER.mkdir(parents=True, exist_ok=True)
settings.METRICS_FOLDER.mkdir(parents=True, exist_ok=True)
