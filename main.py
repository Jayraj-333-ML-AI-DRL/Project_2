from src.MLproject.logger import logging
from src.MLproject.exception import custom_exception 
from src.MLproject.pipelines.training_pipeline import TrainingPipeline
import sys



if __name__ == "__main__":
    logging.info ("The executioin has started")
    
    
    try:
       training_pipline = TrainingPipeline()
       
       r2_score_best_model =  training_pipline.initiate_training()
       
       print(r2_score_best_model)
       
       
       print('Training finished')
       print('Now You can check your model inot Artifacts folder')
        
        
    except Exception as e:
        logging.info ("Custom Exception ")
        raise custom_exception(e,sys)