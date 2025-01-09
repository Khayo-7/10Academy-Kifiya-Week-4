import numpy as np
import pandas as pd

from scripts.utils.logger import setup_logger

# Setup logger for data_cleaning
logger = setup_logger("data_cleaning")

def clean_data(data, date_column = 'Date'):
    
    data = data.copy()
    data = format_date_column(data, date_column)
    
    data = data.drop(columns=['Id']) if 'Id' in data.columns else data
    data ['StateHoliday'] = data['StateHoliday'].astype(str)

    return data

def format_date_column(data, date_column="Date"):
    
    data = data.copy()
    data[date_column] = pd.to_datetime(data[date_column])

    return data

def handle_missing_values(data, strategies):
    """
    Handle missing values in the DataFrame based on strategies provided.
    Args:
        data (pd.DataFrame): DataFrame to clean.
        strategies (dict): Column-wise strategies (e.g., {'col1': 'mean', 'col2': 'fillna_value'})
    Returns:
        pd.DataFrame: Cleaned DataFrame with missing values handled.
    """
    
    data = data.copy()
    for col, strategy in strategies.items():
        if col not in data.columns:
            logger.warning(f"Column {col} not found in DataFrame.")
            continue
        if strategy == "mean":
            data[col] = data[col].fillna(data[col].mean())
            logger.info(f"Filled missing values in {col} with mean.")
        elif strategy == "median":
            data[col] = data[col].fillna(data[col].median())
            logger.info(f"Filled missing values in {col} with median.")
        elif strategy == "mode":
            data[col] = data[col].fillna(data[col].mode()[0])
            logger.info(f"Filled missing values in {col} with mode.")
        elif isinstance(strategy, (int, float, str)):
            data[col] = data[col].fillna(strategy)
            logger.info(f"Filled missing values in {col} with {strategy}.")
        else:
            logger.error(f"Unknown strategy for column {col}: {strategy}")

    return data

def detect_outliers(data, columns):
    """
    Detect outliers using the IQR method.
    Args:
        data (pd.DataFrame): DataFrame to analyze.
        columns (list): List of column names to check for outliers.
    Returns:
        dict: Outliers for each column.
    """
    
    data = data.copy()
    outliers = {}
    for col in columns:
        if col not in data.columns:
            logger.warning(f"Column {col} not found in DataFrame.")
            continue
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers[col] = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
        logger.info(f"Detected {len(outliers[col])} outliers in {col}.")

    return outliers

def remove_irrelevant_columns(data, columns_to_drop):
    """
    Drop irrelevant or redundant columns from the DataFrame.
    Args:
        data (pd.DataFrame): DataFrame to clean.
        columns_to_drop (list): List of column names to drop.
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    
    data = data.copy()
    data = data.drop(columns=columns_to_drop, errors='ignore')
    logger.info(f"Dropped columns: {columns_to_drop}")
    
    return data


def missing_value_summary(data: pd.DataFrame) -> pd.DataFrame:
    """
    Summarizes missing values in each column of the DataFrame.

    Args:
        data (pd.DataFrame): The dataset to evaluate.

    Returns:
        pd.DataFrame: Summary with columns 'Column', 'Missing Values', and 'Percentage'.
    """
    
    # data = data.copy()
    missing = data.isnull().sum()
    percent = (missing / len(data)) * 100
    summary = pd.DataFrame({
        "Column": data.columns,
        "Missing Values": missing,
        "Percentage": percent
    }).reset_index(drop=True)
    return summary.sort_values(by="Missing Values", ascending=False)

