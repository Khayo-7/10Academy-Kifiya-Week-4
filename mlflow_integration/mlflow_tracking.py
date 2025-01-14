import os
import yaml
import mlflow
import mlflow.keras
import mlflow.pyfunc
import mlflow.sklearn
from mlflow import log_params, log_metrics, start_run, end_run

class MLFlowTracking:
    def __init__(self, config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)['mlflow']
        
        mlflow.set_tracking_uri(config['tracking_uri'])
        # mlflow.set_tracking_uri(config['tracking_server_uri']) # For server
        self.experiment_name = config['experiment_name']
        self.default_artifact_root = config['default_artifact_root']
        
        # Set experiment
        # mlflow.create_experiment(self.experiment_name)
        mlflow.set_experiment(self.experiment_name)
    
    def enable_autolog():
        """Enable autolog."""
        mlflow.autolog()

    def start_experiment(self, run_name):
        """Starts a new MLflow run."""
        self.run = start_run(run_name=run_name)

    def log_params(self, params: dict):
        """Logs parameters to MLflow."""
        log_params(params)
    
    def log_metrics(self, metrics: dict):
        """Logs metrics to MLflow."""
        log_metrics(metrics)    
    
    def log_artifact(self, file_path, artifact_path):
        """Logs artifact like plot or result."""
        mlflow.log_artifact(file_path, artifact_path=artifact_path)
    
    def log_artifacts(self, artifact_path):
        """Logs artifacts like plots and results."""
        mlflow.log_artifacts(artifact_path)

    def log_model(self, model, artifact_path, model_name):
        """Log model with keras."""
        mlflow.keras.log_model(model, artifact_path=artifact_path, registered_model_name=model_name)
    
    def log_model_sklearn(self, model, artifact_path, model_name):
        """Log model with sklearn."""
        mlflow.sklearn.log_model(model, artifact_path=artifact_path, registered_model_name=model_name)

    def log_model_pyfunc(self, model, artifact_path, model_name):
        """Log model with pyfunc."""
        model_path = os.path.join(artifact_path, model_name)
        mlflow.pyfunc.save_model(path=model_path, python_model=model)

    def load_model(self, run_id, model_name):
        """Load model with keras."""
        model_uri = f"runs:/{run_id}/{model_name}"
        model = mlflow.keras.load_model(model_uri)
        return model
    
    def load_model_sklearn(self, run_id, model_name):
        """Load model with sklearn."""
        model_uri = f"runs:/{run_id}/{model_name}"
        model = mlflow.sklearn.load_model(model_uri)
        return model
    
    def load_model_pyfunc(self, artifact_path, model_name):
        """Load model with pyfunc."""        
        model_path = os.path.join(artifact_path, model_name)
        model = mlflow.pyfunc.load_model(model_path)
        return model

    def load_latest_model(self, model_name):
        """Load the latest model."""

        # Get the latest run
        client = mlflow.tracking.MlflowClient()
        experiment_id = client.get_experiment_by_name(self.experiment_name).experiment_id
        runs = client.search_runs(experiment_id, order_by=["start_time DESC"], max_results=1)
        latest_run_id = runs[0].info.run_id
        
        # Load the latest model
        model = self.load_model(latest_run_id, model_name)
        return model
    
    def register_model(self, model_name):
        """Register model."""
        RUN_ID = self.get_run_id()
        mlflow.register_model(f"runs:/{RUN_ID}/model", model_name)

    def get_run_id(self):
        """Return Run ID."""
        run_id = self.run.info.run_id
        return run_id
    
    def get_run_data(self):
        """Run data."""
        print(self.run.data.params)
        print(self.run.data.metrics)

    def end_experiment(self):
        """Ends the MLflow run."""
        end_run()

    def get_experiment_list(self):
        """Return list of all experiments."""
        return mlflow.list_experiments()

    def list_experiments(self):
        """List all experiments."""
        experiment_list = self.get_experiment_list()
        for experiment in experiment_list:
            print(experiment)

    # def get_run_(self):
    #     return self.run or mlflow.get_run(self.get_run_id())
    
    # def log_hyperparameter_tuning(self, hyperparameter_space):
    #     for params in hyperparameter_space:
    #         with mlflow.start_run():
    #             # Train the model with parameters
    #             mlflow.log_param(params)