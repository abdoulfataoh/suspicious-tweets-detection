# coding: utf-8

import logging
from pathlib import Path

logging.basicConfig(level=logging.DEBUG)

# [ Dataset ]
DATASET_URL = 'https://drive.google.com/uc?id=1US0luOWPOeVPpUQnpyxr41zrBmeg4Gjk'  # noqa: 501

DATASET_NAME = 'tweets_suspect.csv'

# [ Path settings ]
ASSETS_FOLDER = Path('./assets')
DATASET_FOLDER = ASSETS_FOLDER / 'dataset'
METRICS_FOLDER = ASSETS_FOLDER / 'metrics'
MODELS_FOLDER = ASSETS_FOLDER / 'models'

# [WORD2VEC]
VECTOR_SIZE = 300
