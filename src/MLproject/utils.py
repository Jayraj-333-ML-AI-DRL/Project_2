import  os
import sys
import dill
from src.MLproject.exception import custom_exception
from src.MLproject.logger import logging
import pandas as pd 
from dotenv import load_dotenv
import pymysql

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
import pickle
import numpy 

load_dotenv()

host = os.getenv('host')
user = os.getenv('user')
password= os.getenv('password')
db = os.getenv('db')

def read_sql_data():
    try:
        mydb = pymysql.connect(host=host,
                               user=user,
                               password=password,
                               db=db)
        
        
        logging.info('Connection established')
        
        
        df = pd.read_sql_query('select * from studentsperformance ', mydb)
        
        print(df.head())
        return df
    except Exception as e:
        raise custom_exception(e,sys)


def save_obj(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path,exist_ok=True)
        
        
        with open(file_path,'wb') as f:
            pickle.dump(obj,f)
            
    except Exception as e :
        raise custom_exception(e,sys)  
            
            
def load_obj(file_path):
    try:
        
        with open(file_path,'rb') as obj:
            return dill.load(obj)
               
    except Exception as e:
        raise custom_exception(e,sys)
            
            
    
    
    
    
def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report
    except Exception as e:
        raise custom_exception(e, sys)
    
    