# ml/src/data_ingestion/ingest.py
import pandas as pd
import boto3
from io import StringIO
import os
from dotenv import load_dotenv
import dvc.api
from datetime import datetime
import logging

load_dotenv()

class DataIngestion:
    def __init__(self):
        self.s3_bucket = os.getenv('S3_BUCKET_NAME')
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
        )
        self.logger = logging.getLogger(__name__)
        
    def fetch_raw_data(self, years: list) -> pd.DataFrame:
        """Fetch 10 years of insurance data from S3"""
        self.logger.info(f"Fetching raw data for years: {years}")
        all_data = []
        for year in years:
            try:
                key = f"raw/insurance_data_{year}.csv"
                obj = self.s3_client.get_object(Bucket=self.s3_bucket, Key=key)
                df = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
                df['year'] = year
                all_data.append(df)
                self.logger.info(f"Successfully fetched data for year {year}")
            except Exception as e:
                self.logger.error(f"Error fetching data for year {year}: {str(e)}")
                continue
        
        if not all_data:
            raise ValueError("No data could be fetched")
            
        combined_df = pd.concat(all_data, ignore_index=True)
        self._save_to_dvc(combined_df, "raw_data")
        return combined_df
    
    def _save_to_dvc(self, df: pd.DataFrame, name: str):
        """Save data to DVC tracked directory"""
        path = f"data/raw/{name}_{datetime.now().strftime('%Y%m%d')}.csv"
        df.to_csv(path, index=False)
        self.logger.info(f"Saved raw data to {path}")