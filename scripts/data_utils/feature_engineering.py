import holidays
from scripts.utils.logger import setup_logger

# Setup logger for feature_engineering.py
logger = setup_logger("feature_engineering")

def generate_holiday_data(data, country='ET'):
    """
    This function generates a new column 'is_holiday' in the dataframe based on the country's holidays.

    Args:
        data (pd.DataFrame): The dataframe to add the 'is_holiday' column to.
        country (str, optional): The country code for which to generate the holidays. Defaults to 'ET'.

    Returns:
        None
    """
    logger.info(f"Generating holiday data for {country}...")
    country_holidays = holidays.country_holidays(country)
    data['is_holiday'] = data.index.to_series().apply(lambda date: date in country_holidays).astype(bool)
    logger.info("Holiday data generated successfully.")

def calculate_sales_growth(data):
    """
    This function calculates the sales growth rate.

    Args:
        data (pd.DataFrame): The dataframe containing the sales data.

    Returns:
        pd.DataFrame: The dataframe with the sales growth rate calculated.
    """
    logger.info("Calculating sales growth rate...")
    data['SalesGrowthRate'] = data['Sales'].pct_change() * 100
    sales_growth_data = data[['Date', 'SalesGrowthRate']].dropna()
    logger.info("Sales growth rate calculated successfully.")
    return sales_growth_data
