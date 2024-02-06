import  os
import sys 
from src.MLproject.exception import custom_exception
from src.MLproject.logger import logging
import pandas as pd 
import numpy as np 

from src.MLproject.utils import read_sql_data,save_obj
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


@dataclass
class DataTransformationconfig:
    data_preprocess_obj_file_path = os.path.join("artifacts","preprocess.pkl")
    
    
class dataTransformation:
    def __init__(self) -> None:
        self.data_transformation_config = DataTransformationconfig()
        
    def get_data_transformation_obj(self):
        
        #'Function for data transformation obj '
        try:
            numerical_column = ["writing_score","reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]
                
                      
            logging.info(f'numerical columns : {numerical_column}')
            logging.info(f'categorical columns : {categorical_columns}')
            
            numerical_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("Scaler",StandardScaler())               
                
                
            ])
            
            
            categorical_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy= 'most_frequent' )),
                ("one_hot_encoder", OneHotEncoder()),
                ("Scaler",StandardScaler(with_mean=False))   #no need to use but for normal distribution
            ])
            
            preprocessor = ColumnTransformer([
                
                ('numerical', numerical_pipeline , numerical_column), 
                ('categorical1', categorical_pipeline , categorical_columns ),
            ])
            return preprocessor
        
        
        except Exception as e:
            raise custom_exception(e,sys)
        
    
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            preprocessor_obj = self.get_data_transformation_obj()
            
            
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
            
            logging.info('train and test reading as input')
            
            #seperate the dependent and indepedent variable 
            target_column  = "math_score"
            
            Input_features_train_data = train_data.drop(columns=[target_column],axis=1)
            Target_features_train_data = train_data[target_column]
            
            
            Input_features_test_data = test_data.drop(columns=[target_column],axis=1)
            Target_features_test_data = test_data[target_column]
            
            logging.info('Applying preprocessing on training and testing data')
            
            
            Input_features_train_arr = preprocessor_obj.fit_transform(Input_features_train_data)
            
            Input_features_test_arr = preprocessor_obj.transform(Input_features_test_data)
            
            
            train_array = np.c_[Input_features_train_arr,np.array(Target_features_train_data)]
            
            test_array = np.c_[Input_features_test_arr,np.array(Target_features_test_data)]
            
            
            
            
            logging.info('Saved Preprocessing info')
            
            
            
            save_obj(file_path=self.data_transformation_config.data_preprocess_obj_file_path, obj=preprocessor_obj)
            
            
            return train_array,test_array,self.data_transformation_config
            
                    
            
            
            
        except Exception as e :
            raise custom_exception(e,sys)
         