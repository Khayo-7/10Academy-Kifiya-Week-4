import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# # Utility for basic style setting
# def setup_plot_style():
#     sns.set_theme(style="whitegrid")
#     plt.rcParams.update({"figure.autolayout": True, "figure.figsize": (10, 6)})

# Set a unified style for all plots
sns.set_theme(style="whitegrid")

def plot_distribution(data, column, title=""):
    """
    Plot the distribution of a single column using seaborn.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(data[column], kde=True, color="blue")
    plt.title(title or f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

def compare_numerical_distributions(train, test, column, title=""):
    """
    Compare distributions of a numerical column in the train and test datasets.
    """
    plt.figure(figsize=(12, 6))
    sns.kdeplot(train[column], label="Train", shade=True, color="blue")
    sns.kdeplot(test[column], label="Test", shade=True, color="red")
    plt.title(title or f"Comparison of {column} in Train and Test Sets")
    plt.xlabel(column)
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()

def compare_categoical_distributions(train, test, column, title=""):
    """
    Compare distributions of a categorical column in the train and test datasets.
    """
    plt.figure(figsize=(12, 6))
    sns.countplot(x=column, data=train, label="Train", color='blue', alpha=0.5)
    sns.countplot(x=column, data=test, label="Test", color='red', alpha=0.5)
    plt.title(title or f"Comparison of {column} in Train and Test Sets")
    plt.xlabel(column)
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()

def analyze_correlation(data, columns, title=""):
    """
    Generate a correlation heatmap for selected columns.
    """
    plt.figure(figsize=(12, 8))
    sns.heatmap(data[columns].corr(), annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title(title or "Correlation Heatmap")
    plt.show()

def sales_trends(data, sales_column="Sales"):
    """
    Plot sales trends over time.
    Assumes the dataset has a 'Date' column in datetime format.
    """
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values(by="Date")
    plt.figure(figsize=(14, 7))
    plt.plot(data["Date"], data[sales_column], label="Sales", color="blue")
    plt.title("Sales Trends Over Time")
    plt.xlabel("Date")
    plt.ylabel(sales_column)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def sales_by_store_type(data, store_data, sales_column="Sales"):
    """
    Compare average sales by store type.
    Merges sales and store data on 'Store' key.
    """
    merged = data.merge(store_data, on="Store")
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="StoreType", y=sales_column, data=merged, palette="Set2")
    plt.title("Sales Distribution by Store Type")
    plt.xlabel("Store Type")
    plt.ylabel(sales_column)
    plt.tight_layout()
    plt.show()

def plot_sales_vs_customers(data, figsize=(12, 8)):
    plt.figure(figsize=figsize)
    sns.scatterplot(x='Customers', y='Sales', data=data)
    plt.title('Sales vs. Customers')
    plt.xlabel('Number of Customers')
    plt.ylabel('Sales')
    plt.show()

def plot_sales_heatmap(data, freq='M', figsize=(15, 10)):
    data['Month'] = pd.to_datetime(data['Date']).dt.to_period(freq)
    # monthly_data = data.pivot_table(index='Month', values='Sales', aggfunc='sum')
    monthly_data = data.groupby('Month')['Sales'].sum().values.reshape(-1, 1)
    plt.figure(figsize=figsize)
    sns.heatmap(monthly_data, annot=True, fmt='.0f', cmap='coolwarm')
    plt.title('Monthly Sales Heatmap')
    plt.ylabel('Month')
    plt.xlabel('Sales')
    plt.show()

def plot_cumulative_sales(data, figsize=(12, 8)):
    data['CumulativeSales'] = data['Sales'].cumsum()
    plt.figure(figsize=figsize)
    plt.plot(data['Date'], data['CumulativeSales'])
    plt.title('Cumulative Sales Over Time')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Sales')
    plt.show()

def plot_correlation_matrix(data, figsize=(12, 8)):
    correlation = data.corr()
    plt.figure(figsize=figsize)
    sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', square=True)
    plt.title('Correlation Matrix')
    plt.show()

def plot_rolling_statistics(data, column_name, title, xlabel, ylabel, freq='M', figsize=(15, 10)):
    
    sales = data[column_name].resample(freq).sum()
    rolling_mean = data[column_name].resample(freq).mean()
    rolling_std = data[column_name].resample(freq).std()

    plt.figure(figsize=figsize)
    plt.plot(sales, label='Sales')
    plt.plot(rolling_mean, label='Rolling Mean')
    plt.plot(rolling_std, label='Rolling Std', linestyle='--')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()


def plot_rolling_statistics_2(data, column_name, title, xlabel, ylabel, freq='M', figsize=(15, 10)):

    sales = data[column_name].resample(freq).sum()
    rolling_mean = sales.rolling(windows=12).mean()
    rolling_std = sales.rolling(windows=12).std()
    
    plt.plot(sales.index, sales, label="Sales")
    plt.plot(rolling_mean.rolling_mean, sales, label="12-month Rolling Mean")
    plt.plot(rolling_std.index, rolling_std, label="12-month Rolling Std")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)      
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_ACF_PACF(data, column_name, lags=40, figsize=(15, 10)):
    plt.figure(figsize=figsize)
    plt.subplot(121)
    plot_acf(data[column_name], lags=lags, ax=plt.gca())
    plt.title('Autocorrelation (ACF)')
    plt.subplot(122)
    plot_pacf(data[column_name], lags=lags, ax=plt.gca())
    plt.title('Partial Autocorrelation (PACF)')
    plt.tight_layout()
    plt.show()

def seasonal_decomposition(data, column_name, freq, figsize=(15, 7)):
    decomposition = seasonal_decompose(data[column_name], model='additive', period=freq)
    decomposition.plot()
    plt.gcf().set_size_inches(figsize)
    plt.show()

def plot_sales(data, column_name, freq='W', figsize=(15, 7)):
    aggregated_sales = data[column_name].resample(freq).sum()
    plt.figure(figsize=figsize)
    plt.plot(aggregated_sales, label='Weekly Sales')
    plt.title('Sales Trend Over Time')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.show()

def plot_weekday_sales(data):
    data['Weekday'] = pd.to_datetime(data['Date']).dt.day_name()
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Weekday', y='Sales', data=data, order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    plt.title('Weekday Sales Distribution')
    plt.xlabel('Day of Week')
    plt.ylabel('Sales')
    plt.show()

def plot_promotion_effect(data, freq='M', figsize=(15, 7)):
    data['Month'] = pd.to_datetime(data['Date']).dt.to_period(freq)
    promo_effect = data.groupby('Promo')['Sales'].mean()
    promo_effect.plot(kind='bar', figsize=figsize, color='skyblue', rot=0)
    plt.title('Average Sales by Promotion')
    plt.xlabel('Promotion Status')
    plt.ylabel('Average Sales')
    plt.show()

def plot_holiday_effect(data, figsize=(12, 8)):

    plt.figure(figsize=figsize)
    sns.boxplot(x='StateHoliday', y='Sales', data=data)
    plt.title('Holiday Effect on Sales')
    plt.xlabel('Holiday Type')
    plt.ylabel('Sales')
    plt.show()

def sales_during_holidays(data, holiday_column, sales_column):
    """
    Compare sales during holidays vs. non-holidays.
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=data[holiday_column], y=data[sales_column], palette="Set2")
    plt.title(f"Sales Distribution by {holiday_column}")
    plt.xlabel(holiday_column)
    plt.ylabel(sales_column)
    plt.tight_layout()
    plt.show()



def plot_box(data, column_name, title):

    sns.boxplot(data=data, y=column_name)
    plt.title(title)
    plt.ylabel(column_name)
    plt.tight_layout()
    plt.show()

def plot_hist(data, column_name, title):

    sns.histplot(data[column_name], kde=True, color='blue')
    plt.title(title)
    plt.ylabel(column_name)
    plt.tight_layout()
    plt.show()


# def plot_sales(data, column_name, title, xlabel, ylabel, freq='w', figsize=(15, 7)):
#     """
#     Plots the sales data over time.

#     Args:
#     - data (pd.DataFrame): DataFrame containing the sales data.
#     - column_name (str): Name of the column containing the sales data.
#     - title (str): Title of the plot.
#     - xlabel (str): Label for the x-axis.
#     - ylabel (str): Label for the y-axis.
#     - figsize (tuple): Size of the figure.
#     """
#     logging.info("Plotting sales data...")

#     sales = data[column_name].resample(freq).sum()
    
#     fig , ax = plt.subplots(figsize=figsize)
#     plt.plot(sales.index, sales)
#     # sns.barplot(data)

#     ax.set_title(title)
#     ax.set_xlabel(xlabel)
#     ax.set_ylabel(ylabel)
        
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     return fig
    
    
# def seasonal_decomposition(data, column_name, title, xlabel, ylabel, freq='M', figsize=(15, 7)):

    
#     logging.info("Plotting Monthly decomposition of sales data...")

#     sales = data[column_name].resample(freq).sum()
#     sales_seasonal_decompose = seasonal_decompose(sales, model='additive')

    
#     fig , ax = plt.subplots(figsize=figsize)
#     sales_seasonal_decompose.plot()
#     # sns.barplot(data)

#     ax.set_title(title)
#     ax.set_xlabel(xlabel)
#     ax.set_ylabel(ylabel)
        
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     return fig

# def plot_ACF_PACF(data, column_name, title, xlabel, ylabel, freq='M', figsize=(15, 10)):

#     logging.info()

#     sales =- data[column_name].resample(freq).sum()
#     n_lags = len(sales) // 3
#     acf_values = acf(sales.dropna(), nlags=n_lags)
#     pacf_values = pacf(sales.dropna(), nlags=n_lags)

#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

#     ax1.stem(range(len(acf_values)), acf_values, use_line_colleciton=True)
#     ax1.axhline(y=0, linestyle='--', color='gray')
#     ax1.axhline(y=-1.96/np.sort(len(sales)), linestyle='--', color='gray')
#     ax1.axhline(y=1.96/np.sort(len(sales)), linestyle='--', color='gray')
#     ax1.set_title(title)
#     ax1.set_xlabel(xlabel)
#     ax1.set_ylabel(ylabel)
    
#     ax2.stem(range(len(pacf_values)), pacf_values, use_line_colleciton=True)
#     ax2.axhline(y=0, linestyle='--', color='gray')
#     ax2.axhline(y=-1.96/np.sort(len(sales)), linestyle='--', color='gray')
#     ax2.axhline(y=1.96/np.sort(len(sales)), linestyle='--', color='gray')
#     ax2.set_title("Partial " + title)
#     ax2.set_xlabel(xlabel)
#     ax2.set_ylabel(ylabel)
        
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     return fig

# def plot_holiday_effect(data):

#     data['IsHoliday'] = data['Is_Holiday'] | (df.index.month == 12)
#     holiday_effect = data.groupby('Is_Holiday')['sales'].mean()
#     holiday_effect.plot(kind='bar')

#     plt.xticks(rotation=45)
#     plt.legend()
#     plt.tight_layout()
#     return fig
    
# def plot_promotion_effect(data, freq='M', figsize=(15, 7)):
#     sales = data.groupby([df.inidex.to_oeriod(freq), 'Promo'])['Sales'].mean().unstack()
#     sales.columns = ['No Promo', 'Promo']
    
#     fig , ax = plt.subplots(figsize=figsize)

#     sales(['No Promo', 'Promo']).plot(figsize=figsize)
#     ax.set_title(title)
#     ax.set_xlabel(xlabel)
#     ax.set_ylabel(ylabel)
    
#     plt.xticks(rotation=45)
#     plt.legend()
#     plt.tight_layout()
#     return fig



# #  plot sales vs customer
# # plot sales heatmap
# # plot commulative sales 
# # sales growth rate
# # correlation ananlysis

