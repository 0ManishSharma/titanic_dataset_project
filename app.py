from src.titanic_project.logger import logging
from src.titanic_project.exception import CustomException
from src.titanic_project.components.data_ingestion import DataIngestion
from src.titanic_project.components.data_transformation import DataTransfomer
from src.titanic_project.components.model_tranier import ModelTrainer
import sys

if __name__ =='__main__':
    logging.info('The execution has started')
    try:
        data_ingestion_obj=DataIngestion()
        train_path,test_path = data_ingestion_obj.intiate_data_ingestion()
        data_transformer = DataTransfomer()
        train_arr,test_arr,preprocessor_path=data_transformer.initiate_data_transformer(train_path,test_path)
        # After data transformation
        model_trainer = ModelTrainer()
        model_bundle, test_accuracy = model_trainer.initate_model_trainer(
                    train_arr, test_arr, preprocessor_path
                )
        print(f"Model trained with test accuracy: {test_accuracy}")

    except Exception as e:
        raise CustomException(e,sys)