# ml/src/models/train.py
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import pandas as pd
import logging
from dotenv import load_dotenv
import dvc.api
import os
from datetime import datetime

load_dotenv()

class ModelTrainer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
        
    def train(self):
        experiment_name = "insurance_fraud_detection"
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run():
            # Load data
            data_path = "data/processed/training_data.csv"
            repo = "."
            version = "v1.0"
            
            data_url = dvc.api.get_url(
                path=data_path,
                repo=repo,
                rev=version
            )
            
            df = pd.read_csv(data_url)
            X = df.drop(['is_fraud'], axis=1)
            y = df['is_fraud']
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)
            
            # Log parameters
            params = {
                'n_estimators': 150,
                'max_depth': 10,
                'min_samples_split': 2,
                'random_state': 42,
                'class_weight': 'balanced'
            }
            mlflow.log_params(params)
            
            # Train model
            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            
            accuracy = accuracy_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_proba)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("roc_auc", roc_auc)
            mlflow.log_metric("precision", report['weighted avg']['precision'])
            mlflow.log_metric("recall", report['weighted avg']['recall'])
            mlflow.log_metric("f1_score", report['weighted avg']['f1-score'])
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            self.logger.info(f"Model trained with ROC AUC: {roc_auc:.4f}")
            return model