from catboost import CatBoostClassifier
from .base_model import BaseModel

class CatBoostModel(BaseModel):
    """
    Класс для работы с моделями CatBoost.
    """
    def __init__(self):
        self.model = None

    def load_model(self, model_path):
        """
        Загружает модель CatBoost из заданного пути.

        :param model_path: Строка, представляющая путь к модели.
        """
        self.model = CatBoostClassifier()
        self.model.load_model(model_path)

    def predict(self, data):
        """
        Возвращает предсказания для заданных данных.

        :param data: Данные для предсказания.
        :return: Список предсказаний, каждое из которых представляет собой массив вероятностей классов.
        """
        predictions = self.model.predict_proba(data)
        return predictions