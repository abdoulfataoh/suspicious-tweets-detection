# coding: utf-8

from pathlib import Path
from typing import Callable

import pandas as pd
import gdown


class DataLoader:

    _dataset_path: Path

    def __init__(self, dataset_path: Path) -> None:
        self._dataset_path = dataset_path

    @staticmethod
    def download_from_gdrive(url: str, destination: str, **kwargs):
        gdown.download(
            url=url,
            destination=destination,
            **kwargs
        )

    def load_dataframe_from_csv(self, processor: Callable, **kwargs):
        df = pd.read_csv(self._dataset_path, **kwargs)
        return processor(df)
