import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from app.models.base_model import BaseModel

class HuggingFaceModel(BaseModel):
    """
    Класс для работы с моделями Hugging Face.
    """
    def __init__(self):
        self.tokenizer = None
        self.model = None

    def load_model(self, model_path="bert_1", local=True):
        """
        Загружает модель и токенизатор Hugging Face из заданного пути.

        :param model_path: Строка, представляющая путь к модели.
        :param local: Булевый флаг, указывающий на то, что модель загружается локально.
        """
        if local:
            self.tokenizer = AutoTokenizer.from_pretrained(f'{model_path}_tok', local_files_only=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)

    def predict(self, texts):
        """
        Возвращает предсказания для заданных текстов.

        :param texts: Список текстов для предсказания.
        :return: Список предсказаний, каждое из которых представляет собой массив вероятностей классов.
        """
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return predictions.tolist()
