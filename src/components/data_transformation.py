import os
import sys
from dataclasses import dataclass
from typing import Any
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

@dataclass
class DataTransformationConfig:
    preprocessor_object_file_path = os.path.join('artifact', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self): 
        '''
        this function is responsible for data transformation
        and will create pkl files which will be used for data transformation
        '''
        try:
            numarical_columns = ['writing_score', 'reading_score']
            categorical_columns = ['gender', 'race_ethnicity', 'parental_level_of_education',
                                    'lunch', 'test_preparation_course']
            
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')), # handling the missing value
                    ('scaler', StandardScaler()) # scale the data
                    ]
            )

            cat_pipeline =Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('encoder', OneHotEncoder())
                ]
            )

            logging.info("Numarical columns standard scaling completed")
            logging.info("Categorical columns encoding completed")

            # compine 2 pipelines together

            preprocessor = ColumnTransformer(
                [
                ("numerical_pipeline", num_pipeline, numarical_columns),
                ('categorical_pipelone', cat_pipeline, categorical_columns)
                ]
            )

            logging.info('Coumn Transformer is created')
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)


    def intiate_data_transformation(self, train_path, test_path):
        '''
        this will apply the preprocessing pipelines of train and test data
        '''
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('Reading train and test data is completed')

            logging.info('Obtaining preprocessing object')
            preprocessing_object = self.get_data_transformation_object()
            
            logging.info('Create X and y variables')
            target_column = 'math_score'
            x_train = train_df.drop(columns=[target_column], axis=1)
            y_train = train_df[target_column]

            x_test = test_df.drop(columns=[target_column], axis=1)
            y_test = test_df[target_column]

            logging.info('Applying preprocessing object on train and test data')
            x_train_arr = preprocessing_object.fit_transform(x_train)
            x_test_arr = preprocessing_object.transform(x_test)

            logging.info('Concatenate arrays "x & y" along second axis') #columns
            train_arr = np.c_[x_train_arr, np.array(y_train)]
            test_arr = np.c_[x_test, np.array(y_test)]

            logging.info('Save preprocessing object "pkl file"')

            save_object(
                self.data_transformation_config.preprocessor_object_file_path,
                obj = preprocessing_object
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_object_file_path


            )


        except Exception as e:
            raise CustomException(e, sys)