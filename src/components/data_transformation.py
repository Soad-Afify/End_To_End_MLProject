import os
import sys
from dataclasses import dataclass
from typing import Any
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
#from src.utils import save_object

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

    def create_data_transformation_object(self):
        try:
            pass

        except Exception as e:
            raise CustomException(e, sys)

