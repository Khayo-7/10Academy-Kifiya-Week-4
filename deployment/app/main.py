import os
import sys
from fastapi import FastAPI
from app.prediction import make_prediction
from app.schemas import SalesPredictionInput, SalesPredictionOutput

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

# Setup logger for deployement
from scripts.utils.logger import setup_logger
logger = setup_logger("deployement")

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Welcome to the LSTM Sales Forecasting API! Server is running!"}

@app.post("/predict_sales", response_model=SalesPredictionOutput)
async def predict_sales(input_data: SalesPredictionInput):
    """
    Endpoint to predict sales using the LSTM model.
    """
    input_dict = input_data.model_dump()
    predicted_sales = make_prediction(input_dict)
    return {"predicted_sales": predicted_sales}

# cd deployment/
# uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# http://127.0.0.1:8000/docs

# docker build -t lstm-sales-api .
# docker run -p 8000:8000 lstm-sales-api
