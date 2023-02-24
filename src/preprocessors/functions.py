# coding: utf-8

import logging
from typing import List
from typing import Iterable
from typing import Any

import spacy
from gensim.models import word2vec
import numpy as np

logger = logging.getLogger(__name__)
nlp = spacy.load("en_core_web_sm")


def tokenizer(input: str):
    doc = nlp(input)
    tokens = [token.text for token in doc]
    return tokens


cleaner = lambda tokens: [token.lower() for token in tokens]  # noqa: E731


def avg_words_vectors(
    word2vec: word2vec,
    words_list: List[str],
    vector_size: int
) -> np.ndarray:
    
    vacabulary = word2vec.wv.index2word
    words_vectors_sum = np.zeros(vector_size, dtype=np.float32)
    words_vectors_count = 0
    for word in words_list:
        if word not in vacabulary:
            raise KeyError(f"word '{word}' is not in the wocabulary")
            continue
        else:
            word_vec = word2vec.wv[word]
            words_vectors_count = words_vectors_count + 1
            words_vectors_sum = np.add(words_vectors_sum, word_vec)
    words_vectors_avg = np.divide(words_vectors_sum, words_vectors_count)
    return words_vectors_avg


def vectorize_messages(
    word2vec: word2vec,
    messages_list: List[str],
    vector_size: int
) -> np.ndarray:
    
    sentences_length = len(messages_list)
    message_index = 0
    messages_vectors = np.zeros((sentences_length, vector_size), dtype=np.float32)
    for message in messages_list:
        messages_vectors[message_index] = avg_words_vectors(
            word2vec,
            messages_list,
            vector_size,
        )
        if message_index % 1000 == 0:
            logger.info(f"vectorize_sentences progess: {message_index}/{sentences_length}")
    return messages_vectors
