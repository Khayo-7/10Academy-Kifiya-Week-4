import os
import sys
import joblib
import logging
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

try:
    from scripts.utils.logger import setup_logger
    from scripts.data_utils.loaders import load_csv
    from scripts.data_utils.cleaning import clean_data
    from scripts.modeling.preprocessing import preprocess_data
except ImportError as e:
    logging(f"Import error: {e}. Please check the module path.")

# Setup logger for deployement
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs')
logger = setup_logger("deployement", log_dir)  

method='label'
group_by = 'Store'
resources_dir = os.path.join('resources')
store_filepath = os.path.join(resources_dir, 'store.csv')
scaler_filepath = os.path.join(resources_dir, 'scaler.pkl')
encoder_filepath = os.path.join(resources_dir, 'encoder.pkl')

try:
    # Load previously saved scaler (used during training)
    logger.info("Loading previously saved scaler...")
    scaler = joblib.load(scaler_filepath)
    logger.info("Loading previously saved encoders...")
    encoders = joblib.load(encoder_filepath)
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

    logger.info("Starting preprocessing of input data...")

    # Ensure consistent column ordering
    initial_columns = ['Store', 'DayOfWeek', 'Date', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday']
    drop_columns = ["Date", "DayOfWeek", "Store", "CompetitionOpenSince"]
    final_columns = ['Open', 'Promo', 'StateHoliday','SchoolHoliday',
                     'SalesPerCustomer', 'PromoEffectiveness', 'SalesGrowthRate', 'StoreType', 'Assortment',
                     'CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2',
                     'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval','IsHoliday', 'Year', 'Month', 'Day', 
                     'Week', 'WeekOfYear', 'Weekday', 'IsWeekend',  'BeforeAfterCompetitorOpening',
                     'CompetitionActive', 'Promo2Active', 'CompetitionMonthsOpen', 'Promo2WeeksActive']

    if not all(col in data.columns for col in initial_columns):
        missing_cols = [col for col in initial_columns if col not in data.columns]
        raise ValueError(f"Input data is missing required columns: {missing_cols}")

    logger.info("Loading processed store data...")
    store = load_csv(store_filepath)

    logger.info("Merging data with processed store data...")
    data = data.merge(store, on='Store', how='left')

    logger.info("Cleaning data...")
    data = clean_data(data)

    logger.info("Preprocessing data...")
    data, _, _ = preprocess_data(data, method=method, encoders=encoders, scaler=scaler)
    
    logger.info("Dropping unnecessary columns...")
    data = data.drop(columns=list(set(drop_columns)))
    
    # Reorder columns to match final_columns
    logger.info("Reordering columns...")
    logger.info(data.columns)
    data = data[final_columns]

    # Convert to NumPy array
    logger.info("Converting data to NumPy array...")
    preprocessed_data = np.array(data.values, dtype=np.float32)

    logger.info("Preprocessing complete.")
    return preprocessed_data

