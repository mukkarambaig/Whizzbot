'''
API for the WhizzBot model. The API is built using FastAPI and the WhizzBot model is used to make predictions.
'''

# Importing required libraries
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from whizzbot import WhizzBot
import uvicorn

# Creating FastAPI app
app = FastAPI()

class PredictionRequest(BaseModel):
    ''' Request body for the prediction endpoint '''
    query: str
    temp: float
    top_k: int
    top_p: float
    max_length: int


class DataPathRequest(BaseModel):
    ''' Request body for the set_data_path endpoint '''
    data_path: str


# Dependency Injection
def get_whizzbot():
    ''' Dependency Injection for the WhizzBot model '''
    return WhizzBot()


# Routes

''' Make predictions using the WhizzBot model '''
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


''' Load data into the WhizzBot model '''
@app.get("/load_data")
def load_data(whizzbot: WhizzBot = Depends(get_whizzbot)):
    try:
        whizzbot.load_embeddings()
        return {"Response": "Data loaded"}
    except Exception as e:
        return {"error": str(e)}


''' Set the data path for the WhizzBot model '''
@app.put("/set_data_path")
def set_data_path(request: DataPathRequest, whizzbot: WhizzBot = Depends(get_whizzbot)):
    try:
        whizzbot.set_data_path(request.data_path)
        return {"Response": "Data path updated"}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
