import  os
import sys
import dill
from src.MLproject.exception import custom_exception  # Importing custom exception module
from src.MLproject.logger import logging  # Importing logging module
import pandas as pd 
from dotenv import load_dotenv  # Importing dotenv module to load environment variables
import pymysql  # Importing pymysql for MySQL database connection

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error  # Importing evaluation metrics
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV  # Importing model selection utilities
import pickle  # Importing pickle module for serialization
import numpy  # Importing numpy

load_dotenv()  # Loading environment variables from .env file

host = os.getenv('host')  # Retrieving host from environment variables
user = os.getenv('user')  # Retrieving user from environment variables
password= os.getenv('password')  # Retrieving password from environment variables
db = os.getenv('db')  # Retrieving database name from environment variables

def read_sql_data():
    """
    Function to read data from SQL database.
    """
    try:
        mydb = pymysql.connect(host=host,
                               user=user,
                               password=password,
                               db=db)
        
        logging.info('Connection established')  # Logging successful connection
        
        df = pd.read_sql_query('select * from studentsperformance ', mydb)  # Reading data from SQL database
        
        print(df.head())  # Displaying the first few rows of the dataframe
        return df
    except Exception as e:
        raise custom_exception(e,sys)  # Handling exceptions with custom exception module


def save_obj(file_path, obj):
    """
    Function to save object to a file using pickle serialization.
    """
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path,exist_ok=True)  # Creating directory if not exists
        
        with open(file_path,'wb') as f:
            pickle.dump(obj,f)  # Serializing object and saving to file
    except Exception as e :
        raise custom_exception(e,sys)  # Handling exceptions with custom exception module


def load_obj(file_path):
    """
    Function to load object from file using pickle deserialization.
    """
    try:
        with open(file_path,'rb') as obj:
            return dill.load(obj)  # Deserializing object from file
    except Exception as e:
        raise custom_exception(e,sys)  # Handling exceptions with custom exception module
            
    
def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    """
    Function to evaluate models using specified parameters.
    """
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)  # Performing grid search cross-validation
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)  # Setting best parameters found during grid search
            model.fit(X_train,y_train)  # Training model

            y_train_pred = model.predict(X_train)  # Making predictions on training data
            y_test_pred = model.predict(X_test)  # Making predictions on test data

            train_model_score = r2_score(y_train, y_train_pred)  # Computing R-squared score for training data
            test_model_score = r2_score(y_test, y_test_pred)  # Computing R-squared score for test data

            report[list(models.keys())[i]] = test_model_score  # Storing test model score in report dictionary

        return report  # Returning evaluation report
    except Exception as e:
        raise custom_exception(e, sys)  # Handling exceptions with custom exception module
