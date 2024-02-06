import os
import sys
from src.MLproject.exception import custom_exception
from src.MLproject.logger import logging
import pandas as pd
import numpy as np
from src.MLproject.utils import read_sql_data, save_obj
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

@dataclass
class DataTransformationconfig:
    """Data transformation configuration class."""
    data_preprocess_obj_file_path = os.path.join("artifacts", "preprocess.pkl")

class DataTransformation:
    """Class for data transformation."""

    def __init__(self) -> None:
        """Initialize DataTransformation class."""
        self.data_transformation_config = DataTransformationconfig()

    def get_data_transformation_obj(self):
        """Get data transformation object."""
        try:
            # Define numerical and categorical columns
            numerical_column = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            # Log numerical and categorical columns
            logging.info(f'Numerical columns: {numerical_column}')
            logging.info(f'Categorical columns: {categorical_columns}')

            # Define preprocessing pipelines for numerical and categorical columns
            numerical_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            categorical_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy='most_frequent')),
                ("one_hot_encoder", OneHotEncoder()),
                ("scaler", StandardScaler(with_mean=False))
            ])

            # Define column transformer to apply different preprocessing pipelines to different columns
            preprocessor = ColumnTransformer([
                ('numerical', numerical_pipeline, numerical_column),
                ('categorical', categorical_pipeline, categorical_columns),
            ])

            return preprocessor

        except Exception as e:
            raise custom_exception(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """Initiate data transformation process."""
        try:
            # Get data transformation object
            preprocessor_obj = self.get_data_transformation_obj()

            # Read train and test data
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            # Log reading of train and test data
            logging.info('Train and test data read as input')

            # Separate independent and dependent variables
            target_column = "math_score"
            input_features_train_data = train_data.drop(columns=[target_column], axis=1)
            target_features_train_data = train_data[target_column]
            input_features_test_data = test_data.drop(columns=[target_column], axis=1)
            target_features_test_data = test_data[target_column]

            # Log applying preprocessing on training and testing data
            logging.info('Applying preprocessing on training and testing data')

            # Transform training and testing data using preprocessor object
            input_features_train_arr = preprocessor_obj.fit_transform(input_features_train_data)
            input_features_test_arr = preprocessor_obj.transform(input_features_test_data)

            # Combine features and target into arrays
            train_array = np.c_[input_features_train_arr, np.array(target_features_train_data)]
            test_array = np.c_[input_features_test_arr, np.array(target_features_test_data)]

            # Log saved preprocessing info
            logging.info('Saved preprocessing info')

            # Save preprocessor object
            save_obj(file_path=self.data_transformation_config.data_preprocess_obj_file_path,
                     obj=preprocessor_obj)

            return train_array, test_array, self.data_transformation_config

        except Exception as e:
            raise custom_exception(e, sys)
