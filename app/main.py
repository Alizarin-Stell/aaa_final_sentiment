from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List
from app.models.model_utils import ModelHandler

app = FastAPI()
model_handler = ModelHandler()

DEF_LABELS = ["no emotion", "joy", "sadness", "surprise", "fear", "anger"]


class InputData(BaseModel):
    """
    Класс для ввода данных предсказаний.
    """
    texts: List[str]


class ModelLoadRequest(BaseModel):
    """
    Класс для ввода данных загрузки модели.
    """
    model_type: str
    model_name: str
    local: bool

class LabelRequest(BaseModel):
    """
    Класс для ввода данных меток и текстов.
    """
    texts: List[str]
    labels: List[str] = DEF_LABELS

@app.post("/predict/")
async def predict(input_data: InputData):
    """
    Эндпоинт для получения предсказаний модели.

    :param input_data: Объект, содержащий список текстов.
    :return: Словарь с предсказаниями.
    :raise HTTPException: В случае ошибки.
    """
    try:
        predictions = model_handler.predict(input_data.texts)
        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload_model/")
async def upload_model(model_type: str, file: UploadFile = File(...)):
    """
    Эндпоинт для загрузки модели из бинарного файла.

    :param model_type: Тип модели (`huggingface`).
    :param file: Загрузить файл модели.
    :return: Сообщение о результате загрузки модели.
    :raise HTTPException: В случае ошибки.
    """
    try:
        bin_file_path = f"/tmp/uploaded_model_{file.filename}"
        with open(bin_file_path, "wb") as f:
            f.write(await file.read())

        model_handler.load_model(model_type, bin_file_path, local=True)
        return {"message": f"{model_type} model loaded successfully from binary file"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/load_pretrained_model/")
async def load_pretrained_model(request: ModelLoadRequest):
    """
    Эндпоинт для загрузки предобученной модели или модели из локального пути.

    :param request: Объект, содержащий тип модели и имя предобученной модели или путь к локальной модели.
    :return: Сообщение о результате загрузки модели.
    :raise HTTPException: В случае ошибки.
    """
    try:
        model_handler.load_model(request.model_type, request.model_name, request.local)
        return {"message": f"Model '{request.model_name}' of type '{request.model_type}' loaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/labels/")
async def labels(label_request: LabelRequest):
    """
    Эндпоинт для получения лучших лейблов на основе предсказаний модели.

    :param label_request: Объект, содержащий список текстов и меток.
    :return: Словарь с лейблами.
    :raise HTTPException: В случае ошибки.
    """
    try:
        best_labels = model_handler.get_best_labels(label_request.texts, label_request.labels)
        return {"labels": best_labels}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)