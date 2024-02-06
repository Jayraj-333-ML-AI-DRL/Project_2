import os
import sys
from src.MLproject.logger import logging
from src.MLproject.exception import custom_exception
from src.MLproject.utils import read_sql_data, evaluate_models, save_obj
from dataclasses import dataclass

# Basic imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Modelling
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

@dataclass
class ModelTrainerconfig:
    """Model trainer configuration class."""
    model_trainer_config_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    """Class for training machine learning models."""

    def __init__(self):
        """Initialize ModelTrainer class."""
        self.model_trainer_config = ModelTrainerconfig()

    def initiate_model_trainer(self, train_data, test_data):
        """Initiate model training process."""
        logging.info("Splitting data into arrays.")

        try:
            # Splitting data into features and target variables
            X_train, X_test, y_train, y_test = (train_data[:, :-1], test_data[:, :-1],
                                                 train_data[:, -1], test_data[:, -1])

            # Define models to be evaluated
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            # Define hyperparameters for model tuning
            params = {
                "Decision Tree": {
                    'criterion': ['mse', 'mae', 'friedman_mse', 'poisson'],
                    'splitter': ['best', 'random'],
                },
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting": {
                    'learning_rate': [.1, .01, .05, .001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    'learning_rate': [.1, .01, .05, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "CatBoosting Regressor": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [.1, .01, 0.5, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }
            }

            # Evaluate models and select the best one
            model_report = evaluate_models(X_train, y_train, X_test, y_test, models, params)

            # Get the best model score and name
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            # If the best model's score is less than 0.6, raise an exception
            if best_model_score < 0.6:
                logging.info('No best model found.')
                raise custom_exception("No best model found.")

            # Save the best model
            save_obj(file_path=self.model_trainer_config.model_trainer_config_path, obj=best_model)

            # Make predictions using the best model
            prediction = best_model.predict(X_test)

            # Calculate R2 score
            R2_score = r2_score(y_test, prediction)

            return R2_score

        except Exception as e:
            raise custom_exception(e, sys)
