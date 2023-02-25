# coding: utf-8

import logging

from src import settings
from src.models import Word2vec
from src import DataLoader
from src.preprocessors import tokenizer
from src.preprocessors import avg_words_vectors

logger = logging.getLogger(__name__)


logger.info("download the dataset")
DataLoader.download_from_gdrive(
        url=settings.DATASET_URL,
        destination=settings.DATASET_FOLDER / settings.DATASET_NAME
)

logger.info("load and preprocced the the dataset")
dataloader = DataLoader(settings.DATASET_FOLDER / settings.DATASET_NAME)
dataloader.load_dataframe_from_csv()

dataloader.dataframe['words'] = dataloader.dataframe['message'].progress_apply(lambda x: tokenizer(x.lower()))


logger.info("train word2vec model")
word2vec = Word2vec()
word2vec.train(dataloader.dataframe['words'], vector_size=300)
word2vec.save(settings.MODELS_FOLDER / 'word2vec.model')

logger.info("vactorize dataset")
dataloader.dataframe['avg_words_vectors'] = dataloader.dataframe['words'].progress_apply(
    lambda x: avg_words_vectors(word2vec=word2vec, words_list=x, vector_size=300)
)

dataloader.split_dataframe(
        features_names=['avg_words_vectors'],
        target_name='label',
        test_size=0.2,
        seed=10,
)