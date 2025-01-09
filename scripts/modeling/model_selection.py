from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam


def train_linear_regression(X_train, y_train):
    """
    Train a simple linear regression model.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Target values.
    
    Returns:
        model: Trained Linear Regression model.
    """

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model

def train_random_forest(X_train, y_train, params=None):
    """
    Train Random Forest Regressor with optional hyperparameters.

    Args:
    - X_train: Feature matrix for training.
    - y_train: Target variable.
    - params: (dict) Hyperparameter grid for the Random Forest.

    Returns:
    - model: Trained Random Forest model.
    """
    if not params:
        # Default parameters if none provided
        params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42,
        }

    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    return model
    
def train_xgboost(X_train, y_train, params=None):
    """
    Train an XGBoost Regressor.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Target values.
        params (dict): Hyperparameters for the XGBoost/XGBRegressor.
    
    Returns:
        model: Trained XGBoost model.
    """
    if not params:
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'learning_rate': 0.1,
            'n_estimators': 100,
            'max_depth': 6,
            'verbosity': 0,
            'random_state': 42
        }    

    model = XGBRegressor(**params)
    model.fit(X_train, y_train)

    return model

def train_lightgbm(X_train, y_train, params=None):
    """
    Train a LightGBM model with optional hyperparameters.

    Args:
    - X_train: Feature matrix for training.
    - y_train: Target variable.
    - params: (dict) Hyperparameters for LightGBM.

    Returns:
    - model: Trained LightGBM model.
    """
    if not params:
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'learning_rate': 0.1,
            'n_estimators': 100,
            'random_state': 42
        }

    lgb_model = LGBMRegressor(**params)
    lgb_model.fit(X_train, y_train)
    return lgb_model

def hyperparameter_tuning(model, X_train, y_train, param_grid, choice='grid', n_iter=10):
    """
    Perform hyperparameter tuning for models using RandomizedSearchCV or Grid Search CV.

    Args:
        model: Estimator to tune.
        X_train (pd.DataFrame): Feature matrix for training.
        y_train (pd.Series): Target values.
        param_grid (dict): Grid of hyperparameters.
        choice (str): Type of hyperparameter tuning algorithm.
        n_iter (int): Number of iterations for randomized search.

    Returns:
        best_model: Best model after hyperparameter tuning.
        best_params: Optimal parameters.
    """
    if choice == 'random':
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring='neg_root_mean_squared_error',
            cv=3, # 5
            verbose=1,
            random_state=42,
            n_jobs=-1
        )
    elif choice == 'grid':
        search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    
    search.fit(X_train, y_train)

    return search.best_estimator_, search.best_params_


def train_stacking(models, X_train, y_train):
    """
    Train a stacking regressor with base models.

    Args:
    - models: List of tuples with model names and their instances.
    - X_train: Feature matrix for training.
    - y_train: Target variable.

    Returns:
    - stacked_model: Trained stacking regressor.
    """
    stacked_model = StackingRegressor(
        estimators=models,
        final_estimator=XGBRegressor(),
        # final_estimator=LinearRegression(),
        cv=5,  # 5-fold cross-validation
        n_jobs=-1
    )
    stacked_model.fit(X_train, y_train)
    return stacked_model

def build_lstm_model(input_shape):
    """
    Build an LSTM model for time-series prediction.

    Args:
    - input_shape: Shape of the input data (timesteps, features).

    Returns:
    - model: Compiled LSTM model.
    """
    model = Sequential([
        LSTM(64, activation='relu', return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model
