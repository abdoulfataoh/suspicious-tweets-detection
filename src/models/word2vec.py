# coding: utf-8

from typing import List

from pathlib import Path
from gensim.models import Word2Vec as w2v  # type: ignore
from gensim.models import KeyedVectors

from src.models import Model


class Word2vec(Model):

    _model: w2v
    _vocabulary: KeyedVectors

    def __init__(self) -> None:
        pass

    def train(self, sentences: List[List[str]], vector_size: int, **kwargs):
        self._model = w2v(
            sentences=sentences,
            vector_size=vector_size,
            **kwargs
        )
        self._vocabulary = set(self._model.wv.index_to_key)

    def test(self):
        raise Exception('NotImplemented: test is not implemented')

    def predict(self, inputs: List[List[str]]) -> List:
        predictions = []
        for input in inputs:
            if input in self._vocabulary:
                prediction = self._model.wv.most_similar(input)
                predictions.append(prediction)
            else:
                raise ValueError(f"'{input}' is not in the vocabulary")
        return predictions

    def to_vectors(self, inputs: List[str]) -> List:
        vectors = []
        for input in inputs:
            if input in self._vocabulary:
                vector = self._model.wv[input]
                vectors.append(vector)
            else:
                raise ValueError(f"'{input}' is not in the vocabulary")
        return vectors
    
    def save(self, destination: Path):
        self._model.save(str(destination))
