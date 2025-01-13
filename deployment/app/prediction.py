import os
import sys
import pandas as pd
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError 
from .utils import preprocess_input

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

# Setup logger for deployement
from scripts.utils.logger import setup_logger
logger = setup_logger("deployement")

# Load the trained LSTM model
try:
    logger.info("Starting to load the trained LSTM model...")
    model = load_model(
        "../resources/models/lstm_sales_model.h5",
        custom_objects=None ,
        # custom_objects={"mse": MeanSquaredError()} 
    )
    logger.info("Successfully loaded the trained LSTM model.")
except Exception as e:
    logger.error(f"Error initializing application: {e}")
    raise

def make_prediction(input_data: dict):
    """
    Generate sales prediction using the LSTM model.
    
    Args:
    - input_data: Dictionary of input features.
    
    Returns:
    - predicted_sales: Predicted sales value.
    """
    logger.info("Starting to make prediction...")
    # Convert input to DataFrame for preprocessing
    input_df = pd.DataFrame([input_data])
    processed_input = preprocess_input(input_df) # Scale and encode
    
    # Generate prediction
    prediction = model.predict(processed_input)
    predicted_sales = float(prediction[0][0])
    logger.info("Prediction made successfully.")
    return predicted_sales
