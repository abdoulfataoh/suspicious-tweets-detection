# coding: utf-8

from abc import ABC  # noqa: F401
from abc import abstractclassmethod
from pathlib import Path


class Model(ABC):  # noqa: F811

    def __init__(self, **kwargs):
        pass

    @abstractclassmethod
    def train(self, **kwargs):
        pass

    @abstractclassmethod
    def test(self, **kwargs):
        pass

    @abstractclassmethod
    def save(self, destination: Path):
        pass

    @abstractclassmethod
    def predict(self, **kwargs):
        pass
