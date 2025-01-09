import argparse
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from scripts.data_utils.feature_engineering import *
from scripts.utils.logger import setup_logger

# Setup logger for preprocessing.py
logger = setup_logger("Preprocessing")

def handle_missing_data(data):
    """Handle missing values in the data."""
    data = data.copy()
    logger.info("Starting to handle missing data...")
    # Replace missing CompetitionDistance with a large value (assumed no competition nearby)
    data['CompetitionDistance'] = data['CompetitionDistance'].fillna(data['CompetitionDistance'].max() * 2)

    # Replace missing competition open dates with default values
    data['CompetitionOpenSinceYear'] = data['CompetitionOpenSinceYear'].fillna(data['CompetitionOpenSinceYear'].median())
    data['CompetitionOpenSinceMonth'] = data['CompetitionOpenSinceMonth'].fillna(1)

    # Replace missing Promo2 information with defaults
    data['Promo2SinceYear'] = data['Promo2SinceYear'].fillna(0)
    data['Promo2SinceWeek'] = data['Promo2SinceWeek'].fillna(0)
    data['PromoInterval'] = data['PromoInterval'].fillna('')

    logger.info("Missing data handled successfully.")
    return data

def encode_categorical_features(data, label_encoders=None):
    """Encode categorical variables into numerical values."""

    data = data.copy()
    logger.info("Starting to encode categorical features...")
    if label_encoders is None:
        label_encoders = {}

    for column in ['StoreType', 'Assortment', 'PromoInterval', 'StateHoliday']:
        le = label_encoders.get(column, LabelEncoder())
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

    logger.info("Categorical features encoded successfully.")
    return data, label_encoders

def scale_numerical_features(data, scaler=None):
    """Scale numerical columns for consistency."""
    
    data = data.copy()
    logger.info("Starting to scale numerical features...")
    numerical_columns = ['CompetitionDistance', 'CompetitionMonthsOpen', 'Promo2WeeksActive']

    if scaler is None:
        scaler = StandardScaler()
        data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
    else:
        data[numerical_columns] = scaler.transform(data[numerical_columns])

    logger.info("Numerical features scaled successfully.")
    return data, scaler

def split_data(train_preprocessed, target, drop_columns=None, test_size= 0.2, random_state=42):

    X = train_preprocessed.drop(columns=list(set(drop_columns + [target])))
    y = train_preprocessed[target]
    
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def create_sequences(data, target, timesteps=30):
    """
    Create sequences for LSTM.

    Args:
    - data: Feature matrix (2D array).
    - target: Target variable (1D array).
    - timesteps: Number of previous timesteps to consider.

    Returns:
    - X: 3D array of sequences for LSTM.
    - y: Target variable shifted based on timesteps.
    """
    X, y = [], []
    for i in range(timesteps, len(data)):
        X.append(data[i-timesteps:i])  # Select 'timesteps' rows
        y.append(target[i])           # Target at 'i'
    return np.array(X), np.array(y)


def preprocess_data(data, store, label_encoders=None, scaler=None):
    """Preprocess the data by merging and applying feature engineering."""
    
    data = data.copy()
    logger.info("Starting to preprocess data...")
    # Merge train/test with store data
    data = data.merge(store, on='Store', how='left')

    # Handle Missing Data
    data = handle_missing_data(data)

    # Generate Holidays Features
    data = generate_holiday_data(data)

    # Extract Date Features
    data = extract_date_features(data)

    # Generate Competitor Features
    data = generate_competitor_features(data)

    # Generate Interaction Features
    data = create_interaction_features(data)

    # Encode Categorical Features
    data, label_encoders = encode_categorical_features(data, label_encoders)

    # Scale Numerical Features
    data, scaler = scale_numerical_features(data, scaler)

    logger.info("Data preprocessing completed successfully.")
    return data, label_encoders, scaler

# Main Preprocessing Function
def preprocess_datasets(train, test, store):
    """Preprocess training, test, and store data, applying feature engineering."""

    store_cleaned = handle_missing_data(store)

    # Preprocess train and test for unique cases
    train = generate_sales_features(train)

    # Preprocess training data
    train_preprocessed, label_encoders, scaler = preprocess_data(train, store_cleaned)

    # Preprocess test data using the fitted encoders and scaler from training data
    test_preprocessed, _, _ = preprocess_data(test, store_cleaned, label_encoders, scaler)

    return train_preprocessed, test_preprocessed, store_cleaned