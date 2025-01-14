import os
import joblib
from scripts.modeling.preprocessing  import split_data
from scripts.modeling.model_selection  import build_lstm_model
from mlflow_integration.mlflow_serving import MLFlowServing
from mlflow_integration.mlflow_tracking import MLFlowTracking

resource_dir = os.path.join("..", "resources")
data_path = os.path.join(resource_dir, "data")
models_path = os.path.join(resource_dir, "models")
config_path = os.path.join(resource_dir, "configs")
scalers_path = os.path.join(resource_dir, "scalers")
encoders_path = os.path.join(resource_dir, "encoders")
artifact_path = os.path.join(resource_dir, "model_artifacts")
scaler_file = os.path.join(scalers_path, "scaler.pkl")
config_file = os.path.join(config_path, "mlflow_config.yaml")
store_artifact = os.path.join(data_path, "processed_store.csv")

def load_model(input_shape):

    # Define model
    model = build_lstm_model(
        input_shape,
        learning_rate=params['learning_rate'],
        dropout=params['dropout'],
        loss=params['loss'],
        metrics=params['metrics'],
        opt=params['optimizer'],
    )
    return model

def perform_experiment(model, training_data, params, model_name="sales_model", run_name = "model_train_run"):

    # Initialize MLflow tracking
    mlflow_tracker = MLFlowTracking(config_file)
    
    # Start a new experiment
    mlflow_tracker.start_experiment(run_name=run_name)
    # mlflow_tracker.enable_autolog()    
    
    mlflow_tracker.log_params(params)
    
    # Train the model
    X_train, y_train, X_val, y_val = training_data
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=params['epochs'], batch_size=params['batch_size'])

    # Log metrics
    metrics = {
        "accuracy": 0.93, "loss": 0.12, "mse": 100,
        "final_train_loss": history.history["loss"][-1], 
        "final_val_loss": history.history["val_loss"][-1]
    }
    mlflow_tracker.log_metrics(metrics)
    
    mlflow_tracker.log_artifacts(artifact_path)

    # Log a preprocessor (scaler)
    # joblib.dump(scaler, scaler_file)
    # mlflow_tracker.log_artifact(scaler_file, artifact_path)
    # mlflow_tracker.log_artifact(store_artifact, artifact_path)

    # Log the trained model
    mlflow_tracker.log_model(model, models_path, model_name)
    mlflow_tracker.log_model_sklearn(model, models_path, model_name)
    mlflow_tracker.register_model(model_name=model_name)

    # End the run
    mlflow_tracker.end_experiment()

    model = mlflow_tracker.load_latest_model(model_name=model_name)
    return model


if __name__ == "__main__":

    model_name = "sales_model"
    run_name = "model_train_run"
    # Hyperparameters 
    params = {
        "learning_rate": 0.01, "batch_size": 32, "epochs": 10, 
        "test_size": 0.2, "n_estimators": 100, 'timestep': 2,
        'dropout': 0.2, 'loss': "mse", 'metrics': ["mae"],
        'optimizer': 'adam'
    }
    
    training_data = split_data(X, y)
    input_shape = (training_data[0].shape[1], training_data[0].shape[2])  # (timesteps, features)
    model = load_model(input_shape)

    model = perform_experiment(model, training_data=training_data, params=params, model_name="sales_model", run_name = "model_train_run")
    
    # Test Serving
    MLFlowServing.serve_model(model_uri=f"models:/{model_name}/1", host="127.0.0.1", port=1234)
