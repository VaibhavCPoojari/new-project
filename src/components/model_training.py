import os
import sys
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from src.utils import save_object
from src.utils import evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            x_train, y_train, x_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }
            params = {
                "Random Forest": {
                    'criterion': ['squared_error'],
                    'max_features': ['sqrt'],
                    'n_estimators': [16]
                },
                "Decision Tree": {
                    'criterion': ['squared_error'],
                    'splitter': ['best'],
                    'max_features': ['sqrt'],
                },
                "Gradient Boosting": {
                    'loss': ['squared_error'],
                    'learning_rate': [0.1],
                    'subsample': [0.8],
                    'criterion': ['squared_error'],
                    'max_features': ['sqrt'],
                    'n_estimators': [16]
                },
                "Linear Regression": {},
                "K-Neighbors Regressor": {
                    'n_neighbors': [3],
                    'weights': ['uniform'],
                    'algorithm': ['auto'],
                    'p': [2]
                },
                "XGBRegressor": {
                    'learning_rate': [0.1],
                    'n_estimators': [16]
                },
                "CatBoosting Regressor": {
                    'depth': [6],
                    'learning_rate': [0.1],
                    'iterations': [30]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [0.1],
                    'n_estimators': [16]
                }
            }
            models_reports: dict = evaluate_models(x_train, y_train, x_test, y_test, models, params)
            best_model_score = max(models_reports.values())
            best_model_name = [name for name, score in models_reports.items() if score == best_model_score][0]
            best_model = models[best_model_name]
            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best model found: {best_model_name} with score {best_model_score}")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predicted = best_model.predict(x_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square
        except Exception as e:
            raise CustomException(e, sys)
            
