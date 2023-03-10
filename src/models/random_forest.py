# coding: utf-8

import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support as scores
import numpy

from src.models import Model


class RandomForest(Model):

    _model: RandomForestClassifier

    def __init__(self, **kwargs):
        self._model = RandomForestClassifier(**kwargs)

    def train(self, x_train, y_train):
        self._model.fit(x_train, y_train)

    def test(self, x_test, y_test, **kwargs) -> dict:
        predictions = self.predict(x_test)
        precision, recall, fscore, support = scores(predictions, y_test, **kwargs)  # noqa: 501
        scores_ = {
            'model': str(self),
            'precision': precision,
            'recall': recall,
            'fscore': fscore,
            'support': support,
        }

        return scores_

    def predict(self, inputs: list) -> numpy.ndarray:
        predictions = self._model.predict(inputs)
        return predictions

    def save(self, destination: Path):
        with open(destination, 'wb') as file:
            pickle.dump(self._model, file)

    def __str__(self) -> str:
        return 'RandomForestClassifier'
