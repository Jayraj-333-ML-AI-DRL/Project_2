import os
import sys
from src.MLproject.exception import custom_exception
from src.MLproject.logger import logging
import pandas as pd
from src.MLproject.utils import read_sql_data
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

@dataclass
class DataingestionConfig:
    """Data ingestion configuration class."""
    train_data_path = os.path.join("artifacts", "train.csv")
    test_data_path = os.path.join("artifacts", "test.csv")
    raw_data_path = os.path.join("artifacts", "raw.csv")

class DataIngestion:
    def __init__(self) -> None:
        """Initialize DataIngestion class."""
        self.Ingestion_config = DataingestionConfig()
        
    def initiate_data_ingestion(self):
        """Initiate data ingestion process."""
        try:
            ## Reading data from MySQL ##
            # df = read_sql_data()
            # In case data is read from CSVs
            df = pd.read_csv(os.path.join('Notebook', 'raw.csv'))
            
            logging.info('Reading data completed from MySQL Database')
            
            # Creating necessary directories
            os.makedirs(os.path.dirname(self.Ingestion_config.train_data_path), exist_ok=True)
            
            # Saving raw data to file
            df.to_csv(self.Ingestion_config.raw_data_path, index=False, header=True)
            
            # Splitting data into train and test sets
            train_df, test_df = train_test_split(df, test_size=0.33, random_state=42)
            
            # Saving train and test data to files
            train_df.to_csv(self.Ingestion_config.train_data_path, index=False, header=True)
            test_df.to_csv(self.Ingestion_config.test_data_path, index=False, header=True)
            
            # Logging information about columns
            print(f'all columns name are : {list(df.columns)}')
            logging.info('Train Test split complete')
            logging.info(f'all columns name are : {list(df.columns)}')
            
            return (
                self.Ingestion_config.train_data_path,
                self.Ingestion_config.test_data_path
            )
        except Exception as e:
            # Raise custom exception with error details
            raise custom_exception(e, sys)
