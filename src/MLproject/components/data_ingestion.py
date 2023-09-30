



import  os
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
    train_data_path = os.path.join("artifacts","train.csv")
    test_data_path = os.path.join("artifacts","test.csv")
    raw_data_path = os.path.join("artifacts","raw.csv")
    
    
    
class DataIngestion:
    def __init__(self) -> None:
        self.Ingestion_config = DataingestionConfig()
        
        
    def initiate_data_ingestion(self):
        try :
            ## reading data from mysql ##
            df = read_sql_data()
            
            logging.info('Reading data completed from MySQL Database')
            
            
            os.makedirs(os.path.dirname(self.Ingestion_config.train_data_path),exist_ok= True)
            
            df.to_csv(self.Ingestion_config.raw_data_path,index =False, header = True)
            
            train_df, test_df = train_test_split(df,test_size=0.33, random_state=42)
            
            train_df.to_csv(self.Ingestion_config.train_data_path,index =False, header = True)
            
            test_df.to_csv(self.Ingestion_config.test_data_path,index =False, header = True)
            
            logging.info('Train Test split complete')
            
              
        except Exception as e:
            raise custom_exception(e,sys)