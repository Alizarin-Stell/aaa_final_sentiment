from models.hugging_face_model import HuggingFaceModel
from models.catboost_model import CatBoostModel


class ModelHandler:
    """
    Класс для управления загрузкой моделей и получения предсказаний.
    """

    def __init__(self):
        self.model = None

    def load_model(self, model_type, model_path):
        """
        Загружает модель указанного типа из заданного пути.

        :param model_type: Тип модели (`huggingface` или `catboost`).
        :param model_path: Путь к модели.
        :raise ValueError: Если указан неправильный тип модели.
        """
        if model_type == 'huggingface':
            self.model = HuggingFaceModel()
        elif model_type == 'catboost':
            self.model = CatBoostModel()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        self.model.load_model(model_path)

    def predict(self, data: list):
        """
        Возвращает предсказания для заданных данных.

        :param data: Данные для предсказания.
        :return: Предсказания модели.
        :raise ValueError: Если модель не загружена.
        """
        if self.model is None:
            raise ValueError("No model loaded")
        return self.model.predict(data)

    def get_best_labels(self, data: list, labels: list):
        """
        Получает лучшие метки на основе предсказаний.

        :param data: Данные для предсказания.
        :param labels: Возможные метки.
        :return: Список лучших меток.
        """
        if self.model is None:
            raise ValueError("No model loaded")
        predictions = self.predict(data)
        return self.model.get_best_labels(predictions, labels)