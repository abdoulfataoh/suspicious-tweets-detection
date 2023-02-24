# coding: utf-8

import logging

from src import settings
from src.models import Word2vec
from src import DataLoader
from src.preprocessors import tokenizer
from src.preprocessors import avg_words_vectors
from src.preprocessors import vectorize_messages

logger = logging.getLogger(__name__)


logger.info('download the dataset')
DataLoader.download_from_gdrive(
        url=settings.DATASET_URL,
        destination=settings.DATASET_FOLDER / settings.DATASET_NAME
)

logger.info('load and preprocced the the dataset')
dataloader = DataLoader(settings.DATASET_FOLDER / settings.DATASET_NAME)
dataloader.load_dataframe_from_csv()

dataloader.dataframe['words'] = dataloader.dataframe['message'].progress_apply(lambda x: tokenizer(x.lower()))


logger.info('train word2vec model')
word2vec = Word2vec()
word2vec.train(dataloader.dataframe['words'], vector_size=300)
word2vec.save(settings.MODELS_FOLDER / 'word2vec.model')

print(word2vec.predict([['if']]))