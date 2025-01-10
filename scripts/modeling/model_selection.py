from itertools import product
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow.keras.layers import Bidirectional
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
def build_lstm_model(input_shape, learning_rate=0.001, dropout=0.2, loss='mse', metrics=['mae']):
    """
    Build an LSTM model for time-series prediction with user-defined hyperparameters.

    Args:
    - input_shape: Shape of the input data (timesteps, features).
    - learning_rate: Learning rate for the optimizer.
    - dropout: Dropout rate for the dropout layers.
    - loss: Loss function for the model.
    - metrics: List of metrics to evaluate the model.

    Returns:
    - model: Compiled LSTM model.
    """
    model = Sequential([
        LSTM(64, activation='relu', return_sequences=True, input_shape=input_shape),
        Dropout(dropout),
        LSTM(32, activation='relu'),
        Dropout(dropout),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss, metrics=metrics)
    return model

def build_bidirectional_lstm(input_shape, learning_rate=0.001, dropout=0.2, loss='mse', metrics=['mae']):
    model = Sequential([
        Bidirectional(LSTM(64, activation='relu', return_sequences=True), input_shape=input_shape),
        Dropout(dropout),
        Bidirectional(LSTM(32, activation='relu')),
        Dropout(dropout),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss, metrics=metrics)
    return model
def hyperparameter_tuning_LTSM(X_train, y_train, X_val, y_val, bidirectional=False, config=None):
    """
    Perform hyperparameter tuning for LSTM.

    Args:
    - X_train: Training data (features).
    - y_train: Training data (target).
    - X_val: Validation data (features).
    - y_val: Validation data (target).
    - config: Dictionary of hyperparameter options.

    Returns:
    - best_model: LSTM model with the best validation performance.
    - best_metrics: Evaluation metrics of the best model.
    """
    best_model = None
    best_metrics = {"RMSE": float("inf")}

    if not config:
        config = {
            "units": [32, 64, 128],
            "dropout": [0.1, 0.2, 0.3],
            "learning_rate": [0.001, 0.0005],
            "batch_size": [32, 64],
            "epochs": [50, 100]
        }
    
    param_combinations = list(product(config['units'], config['dropout'], config['learning_rate']))
    
    for units, dropout, lr in param_combinations:
        print(f"Testing configuration: units={units}, dropout={dropout}, lr={lr}")
        model = Sequential()
        if bidirectional:
            model.add(Bidirectional(LSTM(64, activation='relu', return_sequences=True), input_shape=X_train.shape[1:]))
        else:
            model.add(LSTM(units, activation='relu', return_sequences=True, input_shape=X_train.shape[1:]))
        model.add(Dropout(dropout))
        if bidirectional:
            model.add(Bidirectional(LSTM(units // 2, activation='relu')))
        else:
            model.add(LSTM(units // 2, activation='relu'))
        model.add(Dropout(dropout))
        model.add(Dense(1))
        model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
        
        history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val), verbose=0)
        val_rmse = mean_squared_error(y_val, model.predict(X_val), squared=False)

        if val_rmse < best_metrics["RMSE"]:
            best_metrics = {"RMSE": val_rmse}
            best_model = model
            
    return best_model, best_metrics