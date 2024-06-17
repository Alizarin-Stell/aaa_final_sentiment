class BaseModel:
    """
    Базовый класс модели, определяющий интерфейс для всех моделей.
    """
    def load_model(self, model_path: str, local: bool = False):
        """
        Загружает модель из заданного пути.

        :param model_path: Строка, представляющая путь к модели.
        """
        raise NotImplementedError

    def predict(self, data: list):
        """
        Возвращает предсказания для заданных данных.

        :param data: Данные для предсказания.
        :return: Предсказания модели.
        """
        raise NotImplementedError

    def get_best_labels(self, predictions, labels):
        """
        Возвращает лучшие метки на основе предсказаний.

        :param predictions: Список предсказаний.
        :param labels: Список меток, соответствующий индексам предсказаний.
        :return: Список лучших меток.
        """
        best_labels = [labels[pred.index(max(pred))] for pred in predictions]
        return best_labels