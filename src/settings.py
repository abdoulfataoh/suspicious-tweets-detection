# coding: utf-8

import logging
from pathlib import Path

logging.basicConfig(level=logging.DEBUG)

# [ Path settings ]
ASSETS_PATH = Path('./assets')
DATASET_PATH = ASSETS_PATH / 'dataset'
METRICS_PATH = ASSETS_PATH / 'metrics'

