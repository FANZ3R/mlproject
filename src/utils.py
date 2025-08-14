import os
import sys

import pandas as pd
import numpy as np
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_model(x_train, y_train, x_test, y_test, models, params):
    try:
        report = {}
        best_params_report = {}
        best_estimators = {}

        for model_name, model in models.items():
            print(f"Training and tuning {model_name}...")

            param_grid = params.get(model_name, {})
            grid_search = GridSearchCV(model, param_grid, cv=3, scoring="r2", n_jobs=-1)
            grid_search.fit(x_train, y_train)

            best_model = grid_search.best_estimator_
            y_test_pred = best_model.predict(x_test)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score
            best_params_report[model_name] = grid_search.best_params_
            best_estimators[model_name] = best_model   # ✅ save the fitted model

        return report, best_params_report, best_estimators

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)