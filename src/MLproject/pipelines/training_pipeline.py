from src.MLproject.logger import logging  # Importing logging module for logging purposes
from src.MLproject.exception import custom_exception  # Importing custom exception module
from src.MLproject.components.data_ingestion import DataIngestion, DataingestionConfig  # Importing data ingestion components
from src.MLproject.components.data_transformation import DataTransformationconfig, dataTransformation  # Importing data transformation components
from src.MLproject.components.model_trainer import ModelTrainerconfig, ModelTrainer  # Importing model trainer components
import sys,os

class TrainingPipeline:
    """
    Class for training pipeline, which includes data ingestion, transformation, and model training.
    """
    def __init__(self) -> None:
        """
        Constructor method for TrainingPipeline class.
        """
        logging.info('Training pipeline initialized')  # Logging initialization of training pipeline
        # Initializing data ingestion, transformation, and model trainer components
        self.data_ingestion = DataIngestion()
        self.data_transformation = dataTransformation()
        self.model_trainer = ModelTrainer()
    
    def initiate_training(self):
        """
        Method to initiate training pipeline.
        
        Returns:
        float: R-squared score of the trained model.
        """
        try:
            logging.info('Training pipeline executed')  # Logging execution of training pipeline
            
            # Data Ingestion
            train_data_path, test_data_path = self.data_ingestion.initiate_data_ingestion()  # Initiating data ingestion
            
            # Data Transformation
            train_data, test_data, _ = self.data_transformation.initiate_data_transformation(train_data_path, test_data_path)  # Initiating data transformation
            
            # Model Training
            trained_model_R2_score = self.model_trainer.initiate_model_trainer(train_data=train_data, test_data=test_data)  # Initiating model training
            
            # Optionally, you can add evaluation or further steps here
            
            return trained_model_R2_score  # Returning R-squared score of the trained model or any other information
            
        except Exception as e:
            logging.error("Custom Exception occurred")  # Logging custom exception
            raise custom_exception(e, sys)  # Raising custom exception with error information
