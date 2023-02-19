# coding: utf-8


import pytest

import pandas as pd

from src import settings
from src import DataLoader


@pytest.fixture
def dataloader():
    data = \
    """
    surface,qualite,addresse,prix
    100,1,1200,200
    200,2,2400,400
    300,3,3600,600
    100,1,1200,200
    200,2,2400,400
    300,3,3600,600
    100,1,1200,200
    200,2,2400,400
    300,3,3600,600
    100,1,1200,200
    """
    data = data.replace('    ', '')
    dataset_path = settings.DATASET_PATH / 'dataset.test.csv'
    with open(dataset_path, 'w') as file:
        file.write(data)
    
    yield DataLoader(dataset_path)

    dataset_path.resolve().unlink()


def test_load_dataframe_from_csv(dataloader):
    result = {'surface': {0: 100,
                1: 200,
                2: 300,
                3: 100,
                4: 200,
                5: 300,
                6: 100,
                7: 200,
                8: 300,
                9: 100},
                'qualite': {0: 1, 1: 2, 2: 3, 3: 1, 4: 2, 5: 3, 6: 1, 7: 2, 8: 3, 9: 1},
                'addresse': {0: 1200,
                1: 2400,
                2: 3600,
                3: 1200,
                4: 2400,
                5: 3600,
                6: 1200,
                7: 2400,
                8: 3600,
                9: 1200},
                'prix': {0: 200,
                1: 400,
                2: 600,
                3: 200,
                4: 400,
                5: 600,
                6: 200,
                7: 400,
                8: 600,
                9: 200}
            }

    dataloader.load_dataframe_from_csv(lambda x: x)
    assert dataloader.dataframe.to_dict() == result


def test_split_dataframe(dataloader):
    dataloader.load_dataframe_from_csv(lambda x: x)
    x_train, x_test, y_train, y_test = dataloader.split_dataframe(
        label_column_name='prix',
        train_size=0.8,
        seed=10
    )
    
    assert x_train.shape[0] == 8
    assert y_train.shape[0] == 8
    assert x_test.shape[0] == 2
    assert y_test.shape[0] == 2
    assert 'prix' not in x_train.columns
