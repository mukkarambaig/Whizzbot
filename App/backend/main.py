from fastapi import FastAPI, Depends
from pydantic import BaseModel
from whizzbot import WhizzBot
import uvicorn

app = FastAPI()

# Data models
class PredictionRequest(BaseModel):
    query: str
    temp: float
    top_k: int
    top_p: float
    max_length: int


class DataPathRequest(BaseModel):
    data_path: str


# Dependency Injection
def get_whizzbot():
    return WhizzBot()


# Routes
@app.post("/predict")
def make_prediction(request: PredictionRequest, whizzbot: WhizzBot = Depends(get_whizzbot)):
    try:
        response = whizzbot.predict(
            query=request.query,
            temp=request.temp,
            top_k=request.top_k,
            top_p=request.top_p,
            max_length=request.max_length
        )
        return {"prediction": response}
    except Exception as e:
        return {"error": str(e)}


@app.get("/load_data")
def load_data(whizzbot: WhizzBot = Depends(get_whizzbot)):
    try:
        whizzbot.load_embeddings()
        return {"Response": "Data loaded"}
    except Exception as e:
        return {"error": str(e)}


@app.put("/set_data_path")
def set_data_path(request: DataPathRequest, whizzbot: WhizzBot = Depends(get_whizzbot)):
    try:
        whizzbot.set_data_path(request.data_path)
        return {"Response": "Data path updated"}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
