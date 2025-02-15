import os
import pandas as pd
from scripts.utils.logger import setup_logger

# Setup logger for data_loader
logger = setup_logger("data_loader")

def load_csv(file_path: str, sep=',') -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.
    Args:
        file_path (str): Path to the CSV file.
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return None
    
    try:
        logger.info(f"Loading data from file ...")
        data = pd.read_csv(file_path, sep=sep)
        # dataframe = pd.read_csv(file_path, sep='|')
        # dataframe = pd.read_csv(file_path, sep='\t')
        logger.info(f"Loaded data from {file_path}, shape: {data.shape}")

        return data
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return None

def summarize_data(data: pd.DataFrame):
    """Prints summary statistics and info about the dataset."""

    try:
        logger.info("\n--- Data Summary ---")
        logger.info(data.describe())
        logger.info("\n--- Data Info ---")
        logger.info(data.info())
        logger.info("\n--- Total columns with Missing Values ---")
        logger.info((data.isnull().sum() > 0).sum())
        logger.info("\n--- Missing Values ---")
        logger.info(data.isnull().sum())
    except Exception as e:
        logger.error(f"Error summarizing data: {e}")

def save_csv(dataframe, output_path):
    """
    Saves the dataframe to a CSV file.

    Args:
        dataframe (pd.DataFrame): The dataframe to save.
        output_path (str): The path to save the dataframe.

    Returns:
        None
    """
    try:
        dataframe.to_csv(output_path, index=False)
        logger.info(f"Data saved to {output_path} successfully.")
    except Exception as e:
        logger.error(f"Error saving data to {output_path}: {e}")
