
import sys,os
import pandas as pd
from src.MLproject.exception import custom_exception
from src.MLproject.utils import load_obj



class prediction_pipeline:
    def __init__(self) -> None:
        pass
    
    
    def prediction(self,features):
        try:
            print('Load model path and model obj ')
            model_path = os.path.join("artifacts","model.pkl")
            model = load_obj(model_path)
        
        
            print('load preproessor obj path and object')
            preprocessor_obj_path = os.path.join("artifacts","preprocess.pkl")
            preprocessor_obj = load_obj(preprocessor_obj_path)
      
            data_preprocess = preprocessor_obj.transform(features)
            prediction = model.predict(data_preprocess)
            
            
            return prediction
        
        
        
        except Exception as e:
            raise custom_exception(e,sys)
        
        
        
        
class custom_data:
    def __init__( self,
            gender: str,
            race_ethnicity: str,
            parental_level_of_education,
            lunch: str,
            test_preparation_course: str,
            reading_score: int,
            writing_score: int):

            self.gender = gender

            self.race_ethnicity = race_ethnicity

            self.parental_level_of_education = parental_level_of_education

            self.lunch = lunch

            self.test_preparation_course = test_preparation_course

            self.reading_score = reading_score

            self.writing_score = writing_score
                
                
                
                
    def get_data_as_data_frame(self):
            try:
                custom_data_input_dict = {
                        "gender": [self.gender],
                        "race_ethnicity": [self.race_ethnicity],
                        "parental_level_of_education": [self.parental_level_of_education],
                        "lunch": [self.lunch],
                        "test_preparation_course": [self.test_preparation_course],
                        "reading_score": [self.reading_score],
                        "writing_score": [self.writing_score],
                             }

                return pd.DataFrame(custom_data_input_dict)

            except Exception as e:  
                raise custom_exception(e, sys)