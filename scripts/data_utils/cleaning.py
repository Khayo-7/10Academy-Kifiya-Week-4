import os
import logging
import numpy as np
import pandas as pd

from scripts.utils.logger import setup_logger

# Setup logger for data_cleaning
logger = setup_logger("data_cleaning")

def handle_missing_values(df, strategies):
    """
    Handle missing values in the DataFrame based on strategies provided.
    Args:
        df (pd.DataFrame): DataFrame to clean.
        strategies (dict): Column-wise strategies (e.g., {'col1': 'mean', 'col2': 'fillna_value'})
    Returns:
        pd.DataFrame: Cleaned DataFrame with missing values handled.
    """
    for col, strategy in strategies.items():
        if col not in df.columns:
            logger.warning(f"Column {col} not found in DataFrame.")
            continue
        if strategy == "mean":
            df[col].fillna(df[col].mean(), inplace=True)
            logger.info(f"Filled missing values in {col} with mean.")
        elif strategy == "median":
            df[col].fillna(df[col].median(), inplace=True)
            logger.info(f"Filled missing values in {col} with median.")
        elif strategy == "mode":
            df[col].fillna(df[col].mode()[0], inplace=True)
            logger.info(f"Filled missing values in {col} with mode.")
        elif isinstance(strategy, (int, float, str)):
            df[col].fillna(strategy, inplace=True)
            logger.info(f"Filled missing values in {col} with {strategy}.")
        else:
            logger.error(f"Unknown strategy for column {col}: {strategy}")
    return df

def detect_outliers(df, columns):
    """
    Detect outliers using the IQR method.
    Args:
        df (pd.DataFrame): DataFrame to analyze.
        columns (list): List of column names to check for outliers.
    Returns:
        dict: Outliers for each column.
    """
    outliers = {}
    for col in columns:
        if col not in df.columns:
            logger.warning(f"Column {col} not found in DataFrame.")
            continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers[col] = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        logger.info(f"Detected {len(outliers[col])} outliers in {col}.")
    return outliers

def remove_irrelevant_columns(df, columns_to_drop):
    """
    Drop irrelevant or redundant columns from the DataFrame.
    Args:
        df (pd.DataFrame): DataFrame to clean.
        columns_to_drop (list): List of column names to drop.
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    logger.info(f"Dropped columns: {columns_to_drop}")
    return df


def missing_value_summary(data: pd.DataFrame) -> pd.DataFrame:
    """
    Summarizes missing values in each column of the DataFrame.

    Args:
        data (pd.DataFrame): The dataset to evaluate.

    Returns:
        pd.DataFrame: Summary with columns 'Column', 'Missing Values', and 'Percentage'.
    """
    missing = data.isnull().sum()
    percent = (missing / len(data)) * 100
    summary = pd.DataFrame({
        "Column": data.columns,
        "Missing Values": missing,
        "Percentage": percent
    }).reset_index(drop=True)
    return summary.sort_values(by="Missing Values", ascending=False)

