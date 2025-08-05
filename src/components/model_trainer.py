import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,   
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

#from src.components.model_trainer import evaluate_model

# from src.components.model_trainer import ModelTrainerConfig
# from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object
from src.utils import evaluate_model

@dataclass

class ModelTrainerConfig:
    trained_model_file_path: str=os.path.join(
        "artifacts", "model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing input data")
            X_train, y_train, X_test, y_test = (
                    train_array[:,:-1], 
                    train_array[:,-1], 
                    test_array[:,:-1], 
                    test_array[:,-1],
                )
              
            models = {
                    "RandomForestRegressor": RandomForestRegressor(),
                    "DecisionTreeRegressor": DecisionTreeRegressor(),
                    "Gradient Boosting": GradientBoostingRegressor(),
                    "Linear Regression": LinearRegression(),
                    "K-Neighbors Classifier": KNeighborsRegressor(),
                    "XGBClassifier": XGBRegressor(),
                    "CatBoosting Classifier": CatBoostRegressor(verbose=False),
                    "AdaBoost Classifier" : AdaBoostRegressor(),
                }
            
            params = {
                    "DecisionTreeRegressor": {
                        "criterion": ["squared_error", "friedman_mse"],  # default modern options
                        "max_depth": [None, 5, 10]
                    },
                    "RandomForestRegressor": {
                        "n_estimators": [50, 100],
                        "max_depth": [None, 10]
                    },
                    "Gradient Boosting": {
                        "learning_rate": [0.01, 0.1],
                        "n_estimators": [100, 200]
                    },
                    "Linear Regression": {},  # no major params usually tuned
                    "K-Neighbors Classifier": {
                        "n_neighbors": [3, 5, 7]
                    },
                    "XGBClassifier": {
                        "n_estimators": [100, 200],
                        "learning_rate": [0.01, 0.1]
                    },
                    "CatBoosting Classifier": {
                        "depth": [4, 6],
                        "learning_rate": [0.01, 0.1]
                    },
                    "AdaBoost Classifier": {
                        "n_estimators": [50, 100],
                        "learning_rate": [0.01, 0.1]
                    }
                }



            model_report, best_params_report, best_estimators = evaluate_model(
                x_train=X_train,
                y_train=y_train,
                x_test=X_test,
                y_test=y_test,
                models=models,
                params=params
            )
            
            
            
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = best_estimators[best_model_name]

                    
            if best_model_score < 0.6:
                        raise CustomException(
                            "No best model found with sufficient accuracy"
                        )
            logging.info(f"Best model found: {best_model_name} with score: {best_model_score}")    


            save_object(
                        file_path=self.model_trainer_config.trained_model_file_path,
                        obj=best_model
                )
                
            predicted = best_model.predict(X_test)
                
            r2_square = r2_score(y_test,predicted)
            return r2_square
                
                
                
        except Exception as e:
            raise CustomException(e, sys)