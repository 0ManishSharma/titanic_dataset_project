from src.titanic_project.exception import CustomException
from src.titanic_project.logger import logging
import sys
from sklearn.metrics import accuracy_score

import pymysql
from dotenv import load_dotenv
load_dotenv()
import os
import pandas as pd
import pickle


host = os.getenv('host')
user = os.getenv('user')
password = os.getenv('password')
db = os.getenv('database')

def read_sql_data():
    try:
        mydb = pymysql.connect(
            host=host,
            user=user,
            password=password,
            db=db
        )
        logging.info('Database is connected')
        df = pd.read_sql_query('Select * from titanic_dataset',mydb)
        print(df.head())
        
        return df
    except Exception as e:
        raise CustomException(e,sys)
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
def evaluate_model(X_train,y_train,X_test,y_test,models):
    report = {}
    for model_name , model in models.items():
        model.fit(X_train,y_train)
        y_test_pred = model.predict(X_test)
        test_model_score = accuracy_score(y_test,y_test_pred)
        report[model_name] = test_model_score
    return report
    
    
