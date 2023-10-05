import os,sys
from src.MLproject.logger import logging
from src.MLproject.exception import custom_exception

import  os
import sys 
from src.MLproject.exception import custom_exception
from src.MLproject.logger import logging
import pandas as pd 
from src.MLproject.utils import read_sql_data
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
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import warnings


@dataclass
class ModelTrainerconfig:
    self.trained_model_path = os.path.join("artifacts","model.pkl")
    
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerconfig()
        
        
        



