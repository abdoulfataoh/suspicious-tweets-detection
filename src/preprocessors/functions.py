# coding: utf-8

import logging
from typing import List
import re

import spacy
from src.models import Word2vec
import numpy as np

logger = logging.getLogger(__name__)
nlp = spacy.load("en_core_web_sm")


def tokenizer_spacy_en(input: str):
    doc = nlp(input)
    tokens = [token.text for token in doc]
    return tokens


def tokenizer_re(input: str):
    regex = re.compile('\w+')  # noqa: W605
    tokens = re.findall(regex, input)
    return tokens


def avg_words_vectors(
    word2vec: Word2vec,
    words_list: List[str],
    vector_size: int
) -> np.ndarray:

    sum_vectors = np.zeros(vector_size, dtype=np.float32)
    vectors = word2vec.to_vectors(words_list)
    for vector in vectors:
        sum_vectors = np.add(sum_vectors, vector)
    avg_vectors = np.divide(sum_vectors, len(vectors))
    return avg_vectors
