from src.titanic_project.logger import logging
from src.titanic_project.exception import CustomException
from src.titanic_project.components.data_ingestion import DataIngestion
import sys

if __name__ =='__main__':
    logging.info('The execution has started')
    try:
        data_ingestion_obj=DataIngestion()
        train_path,test_path = data_ingestion_obj.intiate_data_ingestion()
    except Exception as e:
        raise CustomException(e,sys)