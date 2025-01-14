import os
import mlflow.pyfunc

class MLFlowServing:
    @staticmethod
    def serve_model(model_uri, host="127.0.0.1", port=1234):
        """Serve a registered model as a REST endpoint."""
        os.system(f"mlflow models serve --model-uri {model_uri} --host {host} --port {port}")
    
    @staticmethod
    def deploy_model(model_path, model_name, version, stage='Staging'):
        """Register and deploy a model to the registry."""
        model = mlflow.pyfunc.load_model(model_path)
        mlflow.register_model(model_uri=model_path, name=model_name)
        
        # Transition to "Staging" or production
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name, version=version, stage=stage
        )