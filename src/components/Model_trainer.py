import os
import sys
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score

from catboost import CatBoostRegressor
from xgboost import XGBRegressor

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifact', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def intiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Split train and test array data')
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1], train_array[:,-1], test_array[:,:-1], test_array[:,-1]
            )

            logging.info("Add models and prams")
            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest" : RandomForestRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "Gradient Bosst": GradientBoostingRegressor(),
                "CatBoost": CatBoostRegressor(),
                "XGB": XGBRegressor()
            }

            params={
                "Linear Regression":{},
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest" : {
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "AdaBoost": {
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Bosst": {
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoost": {
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "XGB": {
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                }
            }

            model_report:dict = evaluate_models(x_train= X_train, y_train=y_train, x_test=X_test, y_test=y_test,
                                                models=models, param=params)
            logging.info("Models report is created")

            # Get best model score
            best_model_score = max(sorted(model_report.values()))

            # Get best model name
            best_model_name = list(model_report.keys())[
                    list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]
            if best_model_score < 0.7:
                raise CustomException("No best model found", sys)
            
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model_name
            )

            logging.info("Best model has been saved")

            predicted=best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return (best_model_name, r2_square)
            
        except Exception as e:
            raise CustomException(e, sys)
