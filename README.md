# aaa_final_sentiment

## Использование Docker
Клонируйте репозиторий:

git clone 
cd {project_root}

Соберите докер образ

docker build -t fastapi-hf-app .
Запуск Docker-контейнера:

docker run -p 8000:8000 fastapi-hf-app

Дальше нужно перейти по ссылке http://localhost:8000/ и ввести один из следующих эндпоинтов


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
Пока что доступно только три варианта, для второго и третьего нужно указывать флаг local = true
Тип пока что тоже только один - huggingface

#### Названия:
- cedr: слабая модель, но она лежит на хагин фейсе
- bert_1: не определяет безэмоциональный текст
- bert_2: определяет безэмоциональный, но из-за этого немного хуже определяет другие

Request:
{
    "model_type": "huggingface",
    "model_name": "cointegrated/rubert-tiny2-cedr-emotion-detection" / "app/bert_1", 
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