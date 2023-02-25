# coding: utf-8

import logging
from typing import List

import spacy
from src.models import Word2vec
import numpy as np

logger = logging.getLogger(__name__)
nlp = spacy.load("en_core_web_sm")


def tokenizer(input: str):
    doc = nlp(input)
    tokens = [token.text for token in doc]
    return tokens


def avg_words_vectors(
    word2vec: Word2vec,
    words_list: List[str],
    vector_size: int
) -> np.ndarray:

    words_vectors_sum = np.zeros(vector_size, dtype=np.float32)
    words_vectors_count = 0
    for word in words_list:
        word_vec = word2vec.to_vectors([word])[0]
        words_vectors_count = words_vectors_count + 1
        words_vectors_sum = np.add(words_vectors_sum, word_vec)
    words_vectors_avg = np.divide(words_vectors_sum, words_vectors_count)
    return words_vectors_avg
