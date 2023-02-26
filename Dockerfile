# Coding: utf-8

# [Base image]
FROM python:3.10

# [Meta data]
LABEL name="suspicious-tweets-detection"
LABEL authors="abdoulfataoh"

# [Wordir]
WORKDIR /app

# [Source code]
ADD ./ /app

# [Install poetry]
RUN pip install -U pip
RUN pip install poetry

# [Install requiered modules]
RUN poetry config virtualenvs.in-project true
RUN poetry install

# [Enable venv]
ENV PATH="/app/.venv/bin:$PATH"

# [Install spacy en_core_web_sm model]
RUN python -m spacy download en_core_web_sm

# [Make test]
RUN make test

# [Train an Test]
RUN ["python", "train_test.py"]