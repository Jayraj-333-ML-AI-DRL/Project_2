from src.MLproject.logger import logging
from src.MLproject.exception import custom_exception 
from src.MLproject.components.data_ingestion import DataIngestion, DataingestionConfig
from src.MLproject.components.data_transformation import DataTransformationconfig, dataTransformation
from src.MLproject.components.model_trainer import ModelTrainerconfig, ModelTrainer
import sys,os

class TrainingPipeline:
    def __init__(self) -> None:
        logging.info('Training pipeline ')
        self.data_ingestion = DataIngestion()
        self.data_transformation = dataTransformation()
        self.model_trainer = ModelTrainer()
    
    def initiate_training(self):
        try:
            logging.info('Training pipline executed')
            # Data Ingestion
            train_data_path, test_data_path = self.data_ingestion.initiate_data_ingestion()
            
            # Data Transformation
            train_data, test_data, _ = self.data_transformation.initiate_data_transformation(train_data_path, test_data_path)
            
            # Model Training
            trained_model_R2_score = self.model_trainer.initiate_model_trainer(train_data=train_data, test_data=test_data)
            
            # Optionally, you can add evaluation or further steps here
            
            return trained_model_R2_score # or any other information you want to return
            
        except Exception as e:
            logging.info ("Custom Exception ")
            raise custom_exception(e,sys)
            
       
