# aaa_final_sentiment

## Использование Docker
Клонируйте репозиторий:

git clone 
cd {project_root}

Соберите докер образ

docker build -t fastapi-hf-app .
Запуск Docker-контейнера:

docker run -p 8000:8000 fastapi-hf-app


## Использование API

### /predict/ [POST]
Получение предсказаний модели.

Request:
{
    "texts": ["Sample text for prediction"]
}

Response:
{
    "predictions": [...]
}


### /load_pretrained_model/ [POST]
Загрузка предобученной модели или локальной модели (тоже предобученой).
Пока что доступно только два варианта, для второго нужно указывать флаг local = true
Тип пока что тоже только один - huggingface

Request:
{
    "model_type": "huggingface",
    "model_name": "cointegrated/rubert-tiny2-cedr-emotion-detection" / "bert_1", 
    "local": true
}

Response:
{
    "message": "Model '...' of type 'huggingface' loaded successfully"
}


### /labels/ [POST]
Получение лучших лейблов на основе предсказаний модели. 
Лейблы расположены в таком порядке ["no emotion", "joy", "sadness", "surprise", "fear", "anger", "mean"]

Request:
{
    "texts": ["Sample text for prediction"],
}

Response:
{
    "labels": ["label2"]
}