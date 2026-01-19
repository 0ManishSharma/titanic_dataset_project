from src.titanic_project.logger import logging
from src.titanic_project.exception import CustomException
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.titanic_project.utils import read_sql_data
import sys
import os
import pandas as pd
@dataclass
class DataIngestionConfig():
    train_data_path = os.path.join('artifacts','train.csv')
    test_data_path = os.path.join('artifacts','test.csv')
    raw_data_path = os.path.join('artifacts','raw.csv')
class DataIngestion():
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
    def intiate_data_ingestion(self):
        try:
            ## reading data from mysql
            logging.info('Read Data from mysql')
            df = pd.read_csv('notebook/Data/raw.csv')
            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.data_ingestion_config.raw_data_path,header=True,index=False)
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.data_ingestion_config.train_data_path,header=True,index=False)
            test_set.to_csv(self.data_ingestion_config.test_data_path,header=True,index=False)
            
            logging.info('Data Ingestion is completed')
            
            return(
                    self.data_ingestion_config.train_data_path,
                    self.data_ingestion_config.test_data_path
                )
        except Exception as e:
            raise CustomException(e,sys)
