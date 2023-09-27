import os
import sys
import pickle
import numpy as np
import pandas as pd
from pymongo import MongoClient
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score,mean_absolute_error
from src.Turbo_Engine_Predict_Maintenance.logger import logging
from src.Turbo_Engine_Predict_Maintenance.exception import CustomException

def connect_to_mongodb(MONGODB_URI, DB_NAME):
    try:
        client = MongoClient(MONGODB_URI)
        db = client[DB_NAME]
        return db
    except Exception as e:
        raise CustomException(e)

def fetch_data_from_mongodb(collection, collection_name):
    try:
        all_documents = list(collection.find())
        return all_documents
    except Exception as e:
        raise CustomException(e)

def remove_id_field(document):
    # Remove the '_id' field if it exists in the document
    if '_id' in document:
        del document['_id']

class RULCalculator:
    def __init__(self, train_df, test_df, rul_df):
        self.train_df = train_df
        self.test_df = test_df
        self.rul_df = rul_df

    def add_rul_to_train_data(self):
        train_grouped_by_unit = self.train_df.groupby(by='engine_number')
        max_time_cycles = train_grouped_by_unit['time_cycles'].max()
        merged = self.train_df.merge(max_time_cycles.to_frame(name='max_time_cycle'), left_on='engine_number', right_index=True)
        merged["RUL"] = merged["max_time_cycle"] - merged['time_cycles']
        merged = merged.drop("max_time_cycle", axis=1)
        return merged

    def add_rul_to_test_data(self):
        test_grouped_by_unit = self.test_df.groupby(by='engine_number')
        max_time_cycles = test_grouped_by_unit['time_cycles'].max()
        merged = self.test_df.merge(max_time_cycles.to_frame(name='max_time_cycle'), left_on='engine_number', right_index=True)
        merged["RUL"] = merged["max_time_cycle"] - merged['time_cycles']
        merged = merged.drop("max_time_cycle", axis=1)
        return merged

    def add_rul_to_test_data_with_rul_df(self):
        test_with_rul = self.add_rul_to_test_data()
        test_with_eolrul = pd.merge(test_with_rul, self.rul_df, on='engine_number', how='left')
        test_with_eolrul['RUL'] = test_with_eolrul['RUL_x'] + test_with_eolrul['RUL_y']
        test_with_eolrul = test_with_eolrul.drop(columns=['RUL_x', 'RUL_y'], axis=1)
        return test_with_eolrul


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        logging.info('Exception occurred in save_object function utils')
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception occurred in load_object function utils')
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3,n_jobs=-1)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)