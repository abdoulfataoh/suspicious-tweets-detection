<p align="center">
  <img src="https://github.com/abdoulfataoh/suspicious-tweets-detection/actions/workflows/train_test.yaml/badge.svg" >
  <img src="https://img.shields.io/badge/best%20model-RandomForestClassifier-red" >
  <img src="https://img.shields.io/badge/precision-99%25-blue" >
  <img src="https://img.shields.io/badge/recall-90%25-yellowgreen" >
  <img src="https://img.shields.io/badge/fscore-94%25-orange" >
</p>

# suspicious-tweets-detection
Machine learning models for suspicious tweets detection

## Dataset
```json
  "dataset_name": "tweets_suspect.csv",
  "dataset_url": "https://drive.google.com/file/d/1US0luOWPOeVPpUQnpyxr41zrBmeg4Gjk/view?usp=share_link",
  "dataset_language": "english",
  "dataset_size": "60_000 rows X 2 Columns",
  "dataset_columns_description":
    {
      "message": "the tweet message",
      "label": "message label, 1 is suspicious, 0 non suspicious" 
    }
```

## Project Workflow
![diagram](https://github.com/abdoulfataoh/suspicious-tweets-detection/blob/main/docs/diagram.png)

## Setup

1. Clone the project
```bash
  git clone git@github.com:abdoulfataoh/suspicious-tweets-detection.git
  cd suspicious-tweets-detection
```

2. Install Dependancies with poetry
```bash
  poetry install
```

3. Enable virtual env
```bash
  poetry shell
```

4. Check everything is OK
```bash
  make test
```

5. Run train and test
```bash
  make train_test
```

## Output
see trained models at ```assets/models```

  
