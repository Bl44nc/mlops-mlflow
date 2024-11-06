from fastapi import FastAPI
from pydantic import BaseModel
from model import load_model, prediction, update_model_back
from torch import tensor
import numpy as np


app = FastAPI()

class Iris(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.post("/predict/")
def predict(data: Iris):
    print(data,"UPUPUPU")
    # tensor_data = tensor([data.sepal_length, data.sepal_width, data.petal_length, data.petal_width])
    array_data = np.array([[
        data.sepal_length, 
        data.sepal_width,
        data.petal_length, 
        data.petal_width
    ]])
    print(array_data, "ARRAY", type(array_data))
    result = prediction(array_data)
    return {"prediction": result[0].item()}

@app.post("/update-model")
def update_model(version: int):
    model_uri = update_model_back(version=version)

    return {"new model": model_uri}
