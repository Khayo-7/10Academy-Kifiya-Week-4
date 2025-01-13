import joblib
import numpy as np
import pandas as pd

from scripts.data_utils.loaders import load_csv
from scripts.modeling.preprocessing import preprocess_data, handle_missing_data

# Load previously saved scaler (used during training)
method='label'
scaler = joblib.load(f"../resources/scalers/{method}_scaler.pkl")
encoders = joblib.load(f"../resources/encoders/{method}_encoders.pkl")

def preprocess_input(data: pd.DataFrame) -> np.ndarray:
    """
    Preprocess raw input for prediction.
    Args:
        data (pd.DataFrame): Input DataFrame containing features.
    Returns:
        np.ndarray: Preprocessed and ready-to-predict input array.
    """

    # Handle missing values
    # (fill missing values with appropriate defaults or use imputations from training)
    numeric_features = ["store", "day_of_week", "promo", "school_holiday"]
    data[numeric_features] = data[numeric_features].fillna(0)  # Default fill for numerics
    categorical_features = ["state_holiday"]
    data[categorical_features] = data[categorical_features].fillna("None")  # Default fill for categoricals
    
    # Scale numerical features
    try:
        data[numeric_features] = scaler.transform(data[numeric_features])
    except ValueError as e:
        raise ValueError(f"Error scaling numerical features. Details: {e}")
    
    # Encode categorical features
    state_holiday_mapping = {"a": 1, "b": 2, "c": 3, "None": 0}
    data["state_holiday"] = data["state_holiday"].map(state_holiday_mapping).fillna(0)
    data = pd.get_dummies(data, columns=["state_holiday"], drop_first=True)
    
    # Ensure consistent column ordering
    final_columns = numeric_features + categorical_features
    if not all(col in data.columns for col in final_columns):
        missing_cols = [col for col in final_columns if col not in data.columns]
        raise ValueError(f"Input data is missing required columns: {missing_cols}")
    
    store = load_csv()
    store_cleaned = handle_missing_data(store)
    data, _, _ = preprocess_data(data, store_cleaned, method=method, label_encoders=encoders, scaler=scaler)

    # Select relevant columns and convert to NumPy array
    preprocessed_data = data[final_columns].values
    return preprocessed_data


