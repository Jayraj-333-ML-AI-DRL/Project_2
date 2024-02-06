from src.MLproject.logger import logging
from src.MLproject.exception import custom_exception 
from src.MLproject.components.data_ingestion import DataIngestion, DataingestionConfig
from src.MLproject.components.data_transformation import DataTransformationconfig, dataTransformation
from src.MLproject.components.model_trainer import ModelTrainerconfig, ModelTrainer

import sys




if __name__=="__main__":
    logging.info("The execution has started")

    try:
        #data_ingestion_config=DataIngestionConfig()
        data_ingestion=DataIngestion()
        train_data_path,test_data_path=data_ingestion.initiate_data_ingestion()

        #data_transformation_config=DataTransformationConfig()
        data_transformation=dataTransformation()
        train_arr,test_arr,_=data_transformation.initiate_data_transormation(train_data_path,test_data_path)

        ## Model Training

        model_trainer=ModelTrainer()
        print(model_trainer.initiate_model_trainer(train_arr,test_arr))
        
    except Exception as e:
        logging.info("Custom Exception")
        raise custom_exception(e,sys)