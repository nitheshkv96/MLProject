import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
import os


# from src.components.data_ingestion import DataIngestion

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifact', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.datatransformation_config = DataTransformationConfig()

        # dataingestObj = DataIngestion()
        # self.raw_pth, self.train_pth, self.test_pth = dataingestObj.initiate_data_ingestion()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        
        '''
        try:
            num_features = ['writing_score', 'reading_score']
            cat_features = ['gender',
                            'race_ethnicity',
                            'parental_level_of_education',
                            'lunch',
                            'test_preparation_course']
            num_pipeline = Pipeline(
                steps = [("imputer", SimpleImputer(strategy = 'median')),
                         ("scaler", StandardScaler())]
            )
            logging.info("Numerical columns standar scaling completed")



            cat_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy = "most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
            logging.info("Categorical columns encoding completed")



            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, num_features ),
                    ("cat_pipeline", cat_pipeline, cat_features )
                ]
            )

            return preprocessor 
        
        except Exception as e:
            raise CustomException(e,sys)


    def initiate_data_tranformation(self, train_pth, test_pth):
        try:
            train_df = pd.read_csv(train_pth)
            test_df = pd.read_csv(test_pth)

            logging.info("Reading train and test data is complete")


            logging.info("Obtaining preprocessing object")

            preprocessor_obj = self.get_data_transformer_object()
            target_feature = "math_score"


            in_feature_train_df = train_df.drop(columns = [target_feature], axis= 1)
            out_feature_train_df = train_df[target_feature]
            in_feature_test_df = test_df.drop(columns = [target_feature], axis= 1)
            out_feature_test_df = test_df[target_feature]

            logging.info(" Applying Preprocessing object on training and test data")

            in_features_train_arr = preprocessor_obj.fit_transform(in_feature_train_df)
            in_features_test_arr = preprocessor_obj.fit_transform(in_feature_test_df)

            train_arr = np.c_[
                in_features_train_arr, np.array(out_feature_train_df)
            ]

            test_arr = np.c_[
                in_features_test_arr, np.array(out_feature_test_df)
            ]

            logging.info("Saved preprocessing Object")
            save_object(
                file_path = self.datatransformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )


            return(
                train_arr,
                test_arr,
                self.datatransformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
