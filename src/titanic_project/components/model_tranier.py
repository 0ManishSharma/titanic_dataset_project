import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from src.titanic_project.logger import logging
from src.titanic_project.exception import CustomException
from src.titanic_project.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initate_model_trainer(self, train_arr, test_arr, preprocessor_path):
        try:
            logging.info('Extracting features and target from train/test arrays')
            
            X_train = train_arr[:, :-1]
            y_train = train_arr[:, -1]
            X_test = test_arr[:, :-1]
            y_test = test_arr[:, -1]
            
            # Define models
            models = {
                'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
                'DecisionTree': DecisionTreeClassifier(random_state=42),
                'RandomForest': RandomForestClassifier(random_state=42),
                'GradientBoosting': GradientBoostingClassifier(random_state=42),
                'SVC': SVC(random_state=42, probability=True)
            }
            
            # Evaluate all models
            model_report = evaluate_model(X_train, y_train, X_test, y_test, models)
            
            # Find best model
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            
            logging.info(f"Best model: {best_model_name} with score: {best_model_score}")
            
            # Hyperparameter tuning
            param_grids = {
                'RandomForest': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7, 10],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'GradientBoosting': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 4, 5],
                    'subsample': [0.8, 0.9, 1.0]
                },
                'LogisticRegression': {
                    'C': [0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear']
                },
                'SVC': {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'rbf'],
                    'gamma': ['scale', 'auto']
                },
                'DecisionTree': {
                    'max_depth': [3, 5, 7, 10],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            }
            
            param_grid = param_grids.get(best_model_name, {})
            
            logging.info(f"Hyperparameter tuning for {best_model_name}")
            random_search = RandomizedSearchCV(
                best_model, param_grid, cv=5, 
                n_iter=20, scoring='accuracy', 
                n_jobs=-1, random_state=42
            )
            
            random_search.fit(X_train, y_train)
            best_model_tuned = random_search.best_estimator_
            
            # Test evaluation
            y_test_pred = best_model_tuned.predict(X_test)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            
            logging.info(f"Tuned model test accuracy: {test_accuracy}")
            logging.info(f"Best params: {random_search.best_params_}")
            
            # Save ONLY the model and preprocessor path
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj={
                    'model': best_model_tuned,
                    'preprocessor_path': preprocessor_path,
                    'model_name': best_model_name
                }
            )
            
            return test_accuracy
            
        except Exception as e:
            raise CustomException(e, sys)
