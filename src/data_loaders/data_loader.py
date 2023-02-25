# coding: utf-8
import logging
from pathlib import Path
from typing import Callable
from typing import List
from typing import Optional
from typing import Union

from tqdm import tqdm
import pandas as pd
import gdown
from sklearn.model_selection import train_test_split

tqdm.pandas()
logger = logging.getLogger(__name__)

__all__ = [
    'DataLoader',
]


class NoneException(Exception):
    def __init__(self, message):
        super.__init__(message)


class NotImplementedException(Exception):
    def __init__(self):
        message = "This feature is currently not implemented"
        super.__init__(message)


class DataLoader:

    _dataset_path: Path
    _dataframe: pd.DataFrame
    _x_train: pd.DataFrame
    _y_train: pd.DataFrame
    _x_test: pd.DataFrame
    _y_test: pd.DataFrame

    def __init__(self, dataset_path: Path) -> None:
        self._dataset_path = dataset_path

    @staticmethod
    def download_from_gdrive(url: str, destination: Path, **kwargs):
        if destination.is_file is True:
            logger.warning('the dataset is already downloaded, we will skip')
        else:
            logger.info('dataset downloading starded...')
            gdown.download(
                url=url,
                output=str(destination),
                **kwargs
            )

    def load_dataframe_from_csv(self, **kwargs):
        df = pd.read_csv(self._dataset_path, **kwargs)
        self._dataframe = df
        return df

    def load_dataframe_from_excel(self, processor: Callable, **kwargs):
        raise NotImplementedException()

    def load_dataframe_from_json(self, processor: Callable, **kwargs):
        raise NotImplementedException()

    def split_dataframe(
            self,
            features_names: List[str],
            target_name: Union[str, int],
            train_size: Optional[float] = None,
            test_size: Optional[float] = None,
            seed: Optional[int] = None,
            **kwargs
    ):
        target = self._dataframe[target_name]
        features = self._dataframe[features_names]

        x_train, x_test, y_train, y_test = train_test_split(
            features,
            target,
            train_size=train_size,
            test_size=test_size,
            random_state=seed,
            **kwargs
        )

        self._x_train = x_train
        self._y_train = y_train
        self._x_test = x_test
        self._y_test = y_test

        return x_train, x_test, y_train, y_test

    @property
    def dataframe(self):
        return self._dataframe

    @property
    def x_train(self):
        if self._x_train is None:
            raise NoneException("split_dataframe must be called before")
        return self._x_train

    @property
    def y_train(self):
        if self._y_train is None:
            raise NoneException("split_dataframe must be called before")
        return self._y_train

    @property
    def x_test(self):
        if self._x_test is None:
            raise NoneException("split_dataframe must be called before")
        return self._x_test

    @property
    def y_test(self):
        if self._y_test is None:
            raise NoneException("split_dataframe must be called before")
        return self._y_test
