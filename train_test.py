# coding: utf-8

import logging

from src import settings
from src.models import Word2vec
from src.models import RandomForest
from src import DataLoader
from src.preprocessors import tokenizer_spacy_en
from src.preprocessors import tokenizer_re
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

dataloader.dataframe['message_words'] = dataloader.dataframe['message'].progress_apply(lambda x: tokenizer_re(x.lower()))
print(dataloader.dataframe['message_words'])
logger.info("train word2vec model")
word2vec = Word2vec()
word2vec.train(
    dataloader.dataframe['message_words'],
    vector_size=300,
    window=5,
    min_count=1,
    workers=4,
)
# word2vec.save(settings.MODELS_FOLDER / 'word2vec.model')
# vs = word2vec.to_vectors(['he', 'that'])
# print(vs)
# print('-------------------')
# print(avg_words_vectors(word2vec, ['he', 'that'], 300))]

logger.info("vactorize dataset")
dataloader.dataframe['avg_words_vectors'] = dataloader.dataframe['message_words'].progress_apply(
    lambda x: avg_words_vectors(word2vec=word2vec, words_list=x, vector_size=300)
)

dataloader.split_dataframe(
        features_names=['avg_words_vectors'],
        target_name='label',
        test_size=0.2,
        seed=42,
        
)
print(dataloader.x_train)
x_train = list(dataloader.x_train['avg_words_vectors'])
x_test = list(dataloader.x_test['avg_words_vectors'])

forest = RandomForest(n_estimators=100)
forest.train(x_train, dataloader.y_train)
result = forest.test(x_test, dataloader.y_test, pos_label=1, average='binary')
print(result)
