import os
import sys
import joblib
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from scripts.utils.logger import setup_logger
from scripts.data_utils.loaders import load_csv
from scripts.data_utils.cleaning import clean_data
from scripts.modeling.preprocessing import preprocess_data

# Setup logger for deployement
logger = setup_logger("deployement")

method='label'
group_by = 'Store'
processed_store_path = "../resources/data/processed_store.csv"

try:
    # Load previously saved scaler (used during training)
    scaler = joblib.load(f"../resources/scalers/{method}_scaler.pkl")
    encoders = joblib.load(f"../resources/encoders/{method}_encoders.pkl")
except Exception as e:
    logger.error(f"Error initializing application: {e}")
    raise

def preprocess_input(data: pd.DataFrame) -> np.ndarray:
    """
    Preprocess raw input for prediction.
    Args:
        data (pd.DataFrame): Input DataFrame containing features.
    Returns:
        np.ndarray: Preprocessed and ready-to-predict input array.
    """

    # Ensure consistent column ordering
    initial_columns = ['Store', 'DayOfWeek', 'Date', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday']
    drop_columns = ["Date", "DayOfWeek", "Store", "CompetitionOpenSince", "SalesGrowthRate"] # 'PromoEffectiveness'
    final_columns = ['Store', 'DayOfWeek', 'Date', 'Open', 'Promo', 'StateHoliday',
                    'SchoolHoliday', 'SalesPerCustomer', 'PromoEffectiveness',
                    'SalesGrowthRate', 'StoreType', 'Assortment', 'CompetitionDistance',
                    'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2',
                    'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval',
                    'CompetitionOpenSince', 'IsHoliday', 'Year', 'Month', 'Day', 'Week',
                    'WeekOfYear', 'Weekday', 'IsWeekend', 'BeforeAfterCompetitorOpening',
                    'CompetitionActive', 'Promo2Active', 'CompetitionMonthsOpen',
                    'Promo2WeeksActive'
                    ]

    if not all(col in data.columns for col in initial_columns):
        missing_cols = [col for col in initial_columns if col not in data.columns]
        raise ValueError(f"Input data is missing required columns: {missing_cols}")
    
    
    data = clean_data(data)
    store = load_csv(processed_store_path)
    # data_preprocessed = data.merge(store, on=group_by, how='inner')
    data, _, _ = preprocess_data(data, store, method=method, label_encoders=encoders, scaler=scaler)
    data = data.drop(columns=list(set(drop_columns)))

    # Reorder columns to match final_columns
    data = data[final_columns]

    # Convert to NumPy array
    preprocessed_data = np.array(data.values, dtype=np.float32)
    return preprocessed_data