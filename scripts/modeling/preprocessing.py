import os
import joblib
import argparse
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from scripts.data_utils.feature_engineering import *
from scripts.data_utils.loaders import save_csv
from scripts.utils.logger import setup_logger

# Setup logger for preprocessing.py
logger = setup_logger("Preprocessing")

resources_dir = os.path.join('..', 'resources')
store_path = os.path.join(resources_dir, 'data')
scaler_path = os.path.join(resources_dir, 'scalers')
encoder_path = os.path.join(resources_dir, 'encoders')

def handle_missing_data(data):
    """
    Handle missing values in the data using imputers.
    """
    data = data.copy()
    logger.info("Starting to handle missing data...")

    data = data.replace([np.inf, -np.inf], np.nan)

    # Create imputers for different strategies
    sales_per_customer_imputer = SimpleImputer(strategy="constant", fill_value=0)
    sales_growth_rate_imputer = SimpleImputer(strategy="constant", fill_value=0)
    open_imputer = SimpleImputer(strategy="constant", fill_value=0)

    # Apply imputers to respective columns
    data['SalesPerCustomer'] = sales_per_customer_imputer.fit_transform(data[['SalesPerCustomer']])
    data['SalesGrowthRate'] = sales_growth_rate_imputer.fit_transform(data[['SalesGrowthRate']])
    data['Open'] = open_imputer.fit_transform(data[['Open']])

    logger.info("Missing data handled successfully using imputers.")
    return data

def handle_missing_store_data(data):
    """
    Handle missing values in the data using imputers for store dataset.
    """
    data = data.copy()
    logger.info("Starting to handle missing data for store dataset...")

    data = data.replace([np.inf, -np.inf], np.nan)

    # Create imputers for different strategies
    competition_distance_imputer = SimpleImputer(strategy="constant", fill_value=data['CompetitionDistance'].max() * 2)
    competition_open_year_imputer = SimpleImputer(strategy="median")
    # competition_open_year_imputer = SimpleImputer(strategy="constant", fill_value=0)
    competition_open_month_imputer = SimpleImputer(strategy="constant", fill_value=1)
    promo2_year_imputer = SimpleImputer(strategy="constant", fill_value=0)
    promo2_week_imputer = SimpleImputer(strategy="constant", fill_value=0)
    promo_interval_imputer = SimpleImputer(strategy="constant", fill_value="NoPromo")

    # Apply imputers to respective columns
    data['CompetitionDistance'] = competition_distance_imputer.fit_transform(data[['CompetitionDistance']])
    data['CompetitionOpenSinceYear'] = competition_open_year_imputer.fit_transform(data[['CompetitionOpenSinceYear']])
    data['CompetitionOpenSinceMonth'] = competition_open_month_imputer.fit_transform(data[['CompetitionOpenSinceMonth']])
    data['Promo2SinceYear'] = promo2_year_imputer.fit_transform(data[['Promo2SinceYear']])
    data['Promo2SinceWeek'] = promo2_week_imputer.fit_transform(data[['Promo2SinceWeek']])
    data['PromoInterval'] = data['PromoInterval'].fillna('NoPromo')
    # data['PromoInterval'] = promo_interval_imputer.fit_transform(data[['PromoInterval']].astype(str))

    logger.info("Missing data handled successfully using imputers for store dataset.")
    return data

def encode_categorical_features(data, method='label', label_encoders=None, onehot_encoder=None, categorical_columns=None):
    """
    Encode categorical variables into numerical values.

    Parameters:
    - data (pd.DataFrame): Input data.
    - method (str): Encoding method. Options: 'label' or 'onehot'.
    - label_encoders (dict): Pre-trained label encoders for label encoding (used with 'label' method).
    - onehot_encoder (ColumnTransformer): Pre-trained ColumnTransformer for one-hot encoding (used with 'onehot' method).
    - categorical_columns (list): List of columns to encode. Defaults to known categorical columns.

    Returns:
    - pd.DataFrame: Encoded data.
    - dict or ColumnTransformer: Updated label encoders (if 'label' method) or fitted ColumnTransformer.
    """
    data = data.copy()
    logger.info("Starting to encode categorical features...")
    
    if categorical_columns is None:
        categorical_columns = ['StoreType', 'Assortment', 'PromoInterval', 'StateHoliday', 'BeforeAfterCompetitorOpening']

    if method not in ['label', 'onehot']:
        raise ValueError("Invalid method. Choose 'label' or 'onehot'.")

    if method == 'label':
        # Use LabelEncoder for each categorical column
        if label_encoders is None:
            label_encoders = {}

        for column in categorical_columns:
            le = label_encoders.get(column, LabelEncoder())
            data[column] = le.fit_transform(data[column].astype(str))
            label_encoders[column] = le

        logger.info("Categorical features encoded with label encoding.")
        return data, label_encoders

    elif method == 'onehot':
        # Use ColumnTransformer with OneHotEncoder
        if onehot_encoder is None:
            onehot_encoder = ColumnTransformer(
                transformers=[
                    ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_columns)
                ],
                remainder='passthrough'  # Keep non-categorical columns as they are
            )

        # Fit and transform the data
        encoded_array = onehot_encoder.fit_transform(data)

        # Extract feature names
        feature_names = onehot_encoder.get_feature_names_out()
        feature_names = [col.replace("onehot__", "").replace("remainder__", "") for col in feature_names]
        encoded_df = pd.DataFrame(encoded_array, columns=feature_names, index=data.index)

        logger.info("Categorical features encoded with one-hot encoding using ColumnTransformer.")
        return encoded_df, onehot_encoder

# def encode_categorical_features(data, label_encoders=None):
#     """Encode categorical variables into numerical values."""

#     data = data.copy()
#     logger.info("Starting to encode categorical features...")
#     if label_encoders is None:
#         label_encoders = {}

#     for column in ['StoreType', 'Assortment', 'PromoInterval', 'StateHoliday']:
#         le = label_encoders.get(column, LabelEncoder())
#         data[column] = le.fit_transform(data[column])
#         label_encoders[column] = le

#     logger.info("Categorical features encoded successfully.")
#     return data, label_encoders

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

def prepare_data_for_modeling(train_data, target, drop_columns=None, sequence=False, timesteps=30):

    X = train_data.drop(columns=list(set(drop_columns + [target])))
    y = train_data[target]
    
    if sequence:
        X, y = create_sequences(X, y, timesteps)

    return X, y

def split_data(X, y, test_size= 0.2, random_state=42):

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    X_val = np.array(X_val, dtype=np.float32)
    y_val = np.array(y_val, dtype=np.float32)

    # X_train = X_train.astype(np.float32)
    # y_train = y_train.astype(np.float32)
    # X_val = X_val.astype(np.float32)
    # y_val = y_val.astype(np.float32)

    return X_train, y_train, X_val, y_val

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

def preprocess_data(data, method='label', encoders=None, scaler=None):
    """Preprocess the data by merging and applying feature engineering."""
    
    data = data.copy()
    logger.info("Starting to preprocess data...")

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
    data, encoders = encode_categorical_features(data, method=method, label_encoders=encoders)

    # Scale Numerical Features
    data, scaler = scale_numerical_features(data, scaler)

    logger.info("Data preprocessing completed successfully.")
    return data, encoders, scaler

# Main Preprocessing Function
def preprocess_datasets(train, test, store, method='label', group_by='Store'):
    """Preprocess training, test, and store data, applying feature engineering."""

    store_cleaned = handle_missing_store_data(store)
    store_preprocessed = generate_competitor_open_date(store_cleaned)

    #  Generate features for training
    train = generate_sales_features(train)

    # Merge train dataset with preprocessed store dataset
    train = train.merge(store_preprocessed, on='Store', how='left')

    # Calculate sales aggregates
    sales_agg = calculate_sales_aggregates(train, group_by=group_by)
    sales_agg = sales_agg.replace([np.inf, -np.inf], np.nan)

    # Merge store data with sales aggregates
    store_processed = store_preprocessed.merge(sales_agg, on=group_by, how='inner')

    # Use store_processed to get aggregated features for the test dataset
    test = test.merge(store_processed, on='Store', how='left')

    # Preprocess training and test data
    train_preprocessed, label_encoders, scaler = preprocess_data(train, method=method)
    test_preprocessed, label_encoders, scaler = preprocess_data(test, method=method, encoders=label_encoders, scaler=scaler)

    # Save the processed store data
    save_csv(store_processed, output_path=os.path.join(store_path, "store_processed.csv"))

    # Save Encoders and Scaler models for a later use in deployment
    joblib.dump(scaler, os.path.join(scaler_path, "scaler.pkl"))
    joblib.dump(label_encoders, os.path.join(encoder_path, f"{method}_encoders.pkl"))

    return train_preprocessed, test_preprocessed, store_processed, scaler, label_encoders