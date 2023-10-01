from src.MLproject.logger import logging
from src.MLproject.exception import custom_exception 
from src.MLproject.components.data_ingestion import DataIngestion,DataingestionConfig
from src.MLproject.components.data_transformation import DataTransformationconfig,dataTransformation
import sys



if __name__ == "__main__":
    logging.info ("The executioin has started")
    
    
    try:
        #data_ingestion_config = DataingestionConfig()
        data_ingestion = DataIngestion()
        train_data_path , test_data_path = data_ingestion.initiate_data_ingestion()
        data_transformation_config = DataTransformationconfig()
        data_transformation = dataTransformation()
        data_transformation.initiate_data_transformation(train_data_path , test_data_path)
        
        
    except Exception as e:
        logging.info ("Custom Exception ")
        raise custom_exception(e,sys)