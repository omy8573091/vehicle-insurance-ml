import mlflow
from mlflow.models.signature import infer_signature
from sklearn.pipeline import make_pipeline

def train_model(X_train, y_train, config):
    """Professional-grade training function with MLflow tracking"""
    mlflow.set_tracking_uri(config['tracking']['uri'])
    mlflow.set_experiment(config['tracking']['experiment'])
    
    with mlflow.start_run(run_name=config['run']['name']) as run:
        # Log all parameters
        mlflow.log_params(config['params'])
        
        # Create model pipeline
        pipeline = make_pipeline(
            Preprocessor(**config['preprocessing']),
            Model(**config['model_params'])
        )
        
        # Train and evaluate
        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(pipeline, X_test, y_test)
        
        # Infer signature
        signature = infer_signature(X_train, pipeline.predict(X_train))
        
        # Log model with metadata
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            signature=signature,
            input_example=X_train[:1],
            registered_model_name=config['model']['name'],
            metadata={
                "framework": "scikit-learn",
                "dataset": config['data']['version']
            }
        )
        
        # Log all metrics
        mlflow.log_metrics(metrics)
        
        # Log artifacts
        mlflow.log_artifact("reports/feature_importance.png")
        mlflow.log_dict(config, "run_config.yaml")
        
        return run.info.run_id