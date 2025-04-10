# ml/src/feature_engineering/features.py
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from datetime import datetime
import logging

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        self.logger.info("Starting feature engineering")
        X = X.copy()
        
        # Convert dates
        X['policy_start_date'] = pd.to_datetime(X['policy_start_date'])
        X['policy_end_date'] = pd.to_datetime(X['policy_end_date'])
        
        # Policy duration features
        X['policy_duration_days'] = (X['policy_end_date'] - X['policy_start_date']).dt.days
        X['policy_active'] = (X['policy_end_date'] > datetime.now()).astype(int)
        
        # Claim frequency
        X['claim_frequency'] = X['previous_claims'] / X['years_with_company']
        X['claim_frequency'].replace([np.inf, -np.inf], 0, inplace=True)
        
        # Risk features
        X['risk_score'] = (
            0.3 * X['previous_claims'] +
            0.2 * X['vehicle_age'] +
            0.1 * X['driver_age'] +
            0.4 * X['claim_frequency']
        )
        
        # Vehicle features
        X['vehicle_age'] = datetime.now().year - X['manufacture_year']
        X['is_old_vehicle'] = (X['vehicle_age'] > 10).astype(int)
        
        self.logger.info("Feature engineering completed")
        return X