import mlflow




mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
model_uri = "models:/iris_model/1"




def load_model(model_uri):
    model = mlflow.pyfunc.load_model(model_uri)
    return model

def prediction(data):
    result = model.predict(data)
    return result

def update_model_back(version: int):
    global model
    model_uri = f"models:/iris_model/{version}"
    model = load_model(model_uri=model_uri)
    return model_uri


model = load_model(model_uri=model_uri)