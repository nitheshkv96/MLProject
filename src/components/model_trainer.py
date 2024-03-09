
import numpy as np
import pandas as pd
import seaborn as sns
# import warnings
# warnings.filterwarnings("ignore")


from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, 
                                AdaBoostRegressor,
                                GradientBoostingRegressor)
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from src.utils import save_object, evaluate_model
import os

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifact','model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("SPlitting Training and Test input data")

            X_train, y_train, X_test, y_test = (
                    train_arr[:,:-1],
                    train_arr[:,-1],
                    test_arr[:,:-1],
                    test_arr[:,-1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Classifier": KNeighborsRegressor(),
                "XGBClassifier": XGBRegressor(),
                "CatBoosting Classifier": CatBoostRegressor(verbose = False),
                "AdaBoost Classifier": CatBoostRegressor(),
            }

            logging.info("Evaluating model on Training and Test input data")
            model_report:dict = evaluate_model(X_train = X_train, 
                                               y_train = y_train,
                                               X_test = X_test,
                                               y_test = y_test,
                                               models = models)
            
            logging.info("Debubbing")
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            print(best_model_name)
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best Model found")
            

            logging.info("Best found model on both training and testing dataset")
            
            save_object(file_path=self.model_trainer_config.trained_model_file_path,
                        obj = best_model)

            predicted = best_model.predict(X_test)


            logging.info("Generating a test score with best model")
            test_score = r2_score(predicted, y_test)

            return test_score 


        except:
            pass

if __name__ == "__main__":
    obj = DataIngestion()
    train_pth, test_pth, _ = obj.initiate_data_ingestion()
    data_tranform_obj = DataTransformation()
    train_arr, test_arr,_ = data_tranform_obj.initiate_data_tranformation(train_pth, test_pth)

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr, test_arr))
