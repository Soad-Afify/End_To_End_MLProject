import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.logger import LOG_FILE_PATH
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.Model_trainer import ModelTrainer
from src.components.Model_trainer import ModelTrainerConfig


@dataclass # no need to use __init__ if we need to define variables, but we if we will create methods in the class it's better to use __inut__
class DataIngestionConfig:
    train_data_path: str= os.path.join('artifact', "train_data.csv")
    test_data_path: str= os.path.join('artifact', "test_data.csv")
    raw_data_path: str= os.path.join('artifact', "raw_data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig() # this attribute will save the data in the above pathes

    def initiate_data_ingestion(self):
        '''
        This function reads the data from data source and create the path for artifact
        '''

        logging.info('Entered the data ingestion method')
        try:
            df = pd.read_csv('Notebook/data/stud.csv')
            logging.info('Data has been loaded in dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            logging.info('Train test split is intiated')
            train_set, test_set =train_test_split(df,test_size=0.2, random_state=42)

            logging.info('Save the data in artifact')
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('Data ingestion is done')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
                )
            
        except Exception as e:
            raise CustomException(e, sys)
        


if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    
    data_transformation = DataTransformation()
    train_arr, test_arr,_ = data_transformation.intiate_data_transformation(train_data, test_data)

    model_trainer = ModelTrainer()
    model_name, score = model_trainer.intiate_model_trainer(train_array=train_arr, test_array=test_arr)
    print("Best model name: ", model_name, " with r2score ", score)
