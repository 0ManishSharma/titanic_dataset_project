from src.titanic_project.exception import CustomException
from src.titanic_project.logger import logging
import sys

import pymysql
from dotenv import load_dotenv
load_dotenv()
import os
import pandas as pd


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

