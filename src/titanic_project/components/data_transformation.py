import sys
from src.titanic_project.logger import logging
from src.titanic_project.exception import CustomException
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from dataclasses import dataclass
from sklearn.preprocessing import LabelEncoder
from src.titanic_project.utils import save_object
import os
import numpy as np
import pandas as pd

@dataclass
class DataTransfromerConfig():
    preprocessor_file_path = os.path.join('artifacts','preprocessor.pkl')
class DataTransfomer():
    def __init__(self):
        self.data_preprocessor_config = DataTransfromerConfig()
    def get_transformer_obj(self):
        numerical_columns =['Pclass','Age','Fare','FamilySize','IsAlone']
        categorical_columns = ['Embarked','AgeGroup']
        
        num_pipeline = Pipeline(steps=[
            ('impute',SimpleImputer(strategy='median')),
            ('scaler',StandardScaler())
        ])
        cat_pipeline=Pipeline(steps=[
            ('impute',SimpleImputer(strategy='most_frequent')),
            ('ohe',OneHotEncoder(drop='first',handle_unknown='ignore'))
        ])
        preprocessor = ColumnTransformer([
            ('num',num_pipeline,numerical_columns),
            ('cat',cat_pipeline,categorical_columns)
        ])
        return preprocessor
    def initiate_data_transformer(self,train_path,test_path):
        
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            for df in [train_df,test_df]:
                df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
                df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
                df['IsAlone'] = 0
                df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1
                df['AgeGroup'] = pd.cut(
                    df['Age'],
                    bins=[0,12,20,40,60,80],
                    labels=['Child','Teen','Adult','MiddleAge','Senior']
                )
                df['Fare'] = np.log1p(df['Fare'])
                le = LabelEncoder()
                df['Sex'] = le.fit_transform(df['Sex'])
            target_column_name=['Survived']
            drop_columns = ['SibSp','Parch','PassengerId','Cabin','Name','Ticket']
            print(train_df.columns)
            print(train_df.columns)
            
            X_train = train_df.drop(columns=target_column_name + drop_columns,errors='ignore')
            y_train = train_df[target_column_name]
            
            X_test = test_df.drop(columns=target_column_name+drop_columns,errors='ignore')
            
            y_test = test_df[target_column_name]
            
            logging.info('Applying preprocessing obj on train and test data')
            
            preprocessing_obj = self.get_transformer_obj()
            
            X_train_arr = preprocessing_obj.fit_transform(X_train)
            
            X_test_arr = preprocessing_obj.transform(X_test)
            
            train_arr = np.c_[
                X_train_arr,
                np.array(y_train)
            ]
            test_arr = np.c_[
                X_test_arr,np.array(y_test)
            ]
            print(test_arr.shape)
            print(train_arr.shape)
            logging.info('Saving preprocessor file')
            save_object(
                file_path = self.data_preprocessor_config.preprocessor_file_path,
                obj = preprocessing_obj
            )
            return(
                train_arr,
                test_arr,
                self.data_preprocessor_config.preprocessor_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)