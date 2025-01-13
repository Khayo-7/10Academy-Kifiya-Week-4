import os
import sys
from fastapi import FastAPI
from app.prediction import make_prediction
from app.schemas import SalesPredictionInput, SalesPredictionOutput

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

# Setup logger for deployement
from scripts.utils.logger import setup_logger
logger = setup_logger("fastapi_deployement")

logger.info("Starting process...")

app = FastAPI()

@app.get("/")
async def root():
    logger.info("Starting root function...")
    response = {"message": "Welcome to the LSTM Sales Forecasting API! Server is running!"}
    logger.info("Ending root function...")
    return response

@app.post("/predict_sales", response_model=SalesPredictionOutput)
async def predict_sales(input_data: SalesPredictionInput):
    logger.info("Starting predict_sales function...")
    """
    Endpoint to predict sales using the LSTM model.
    """
    input_dict = input_data.model_dump()
    predicted_sales = make_prediction(input_dict)
    logger.info("Ending predict_sales function...")
    return {"predicted_sales": predicted_sales}

logger.info("Ending process...")

# uvicorn app.main:app --reload --host 0.0.0.0 --port 7777

# docker build -t lstm-sales-api .
# docker run -p 8000:7777 lstm-sales-api
