import os,sys
from src.MLproject.logger import logging
from src.MLproject.exception import custom_exception

import  os
import sys 
from src.MLproject.exception import custom_exception
from src.MLproject.logger import logging
import pandas as pd 
from src.MLproject.utils import read_sql_data,evaluate_models,save_obj
from dataclasses import dataclass
# Basic Import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
# Modelling
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

@dataclass
class ModelTrainerconfig:
        model_trainer_config_path = os.path.join("artifacts","model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerconfig()
        
    def initiate_model_trainer(self,train_data,test_data):
        logging.info("spliting data into arrys ")
        
        try :
            X_train,X_test, y_train,y_test = (train_data[:,:-1 ],test_data[:,:-1], train_data[:,-1],test_data[:,-1])
            
            
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            
                }
            
            
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }
            
            model_report = evaluate_models(X_train, y_train,X_test,y_test,models,params)
            
            
            #get best model score
            best_model_score = max(sorted(model_report.values()))
            
            best_model_name = list(model_report.keys())[list (model_report.values()).index(best_model_score)]
            
            
            best_model = models[best_model_name]
            
            if best_model_score < 0.6:
                logging.info('no best model found ')
                raise custom_exception("no best model found")
            
            
            save_obj(file_path=self.model_trainer_config.model_trainer_config_path,obj=best_model)
            
            
            prediction = best_model.predict(X_test)
            
            
            R2_score = r2_score(y_test,prediction)
            
            
            return R2_score
            
            
            
            
            
            
        except Exception as e:
            raise custom_exception(e,sys)
        
        
        
        
        
        
        
        
        
        
        



