from src.MLproject.logger import logging
from src.MLproject.exception import custom_exception 
from src.MLproject.components.data_ingestion import DataIngestion,DataingestionConfig

import sys



if __name__ == "__main__":
    logging.info ("The executioin has started")
    
    
    try:
        #data_ingestion_config = DataingestionConfig()
        data_ingestion = DataIngestion()
        data_ingestion.initiate_data_ingestion()
    except Exception as e:
        logging.info ("Custom Exception ")
        raise custom_exception(e,sys)