import os
import sys
import mlflow
from src.Turbo_Engine_Predict_Maintenance.logger import logging
from src.Turbo_Engine_Predict_Maintenance.exception import CustomException
import pandas as pd
from src.config import MONGODB_URI, DB_NAME, COLLECTION_NAMES
from src.Turbo_Engine_Predict_Maintenance.components.data_ingestion import DataIngestion
from src.Turbo_Engine_Predict_Maintenance.components.data_transformation import DataTransformation
from src.Turbo_Engine_Predict_Maintenance.components.model_trainer import ModelTrainer

if __name__ == '__main__':
    mlflow.set_experiment("TrainingPipeline")
    with mlflow.start_run(nested=True):
        obj = DataIngestion(MONGODB_URI, DB_NAME, COLLECTION_NAMES)
        train_data_path, test_data_path, rul_data_path = obj.initiate_data_ingestion()
        mlflow.log_artifact(train_data_path)
        mlflow.log_artifact(test_data_path)
        mlflow.log_artifact(rul_data_path)

        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path, rul_data_path)
        #mlflow.log_artifact(train_arr)
        #mlflow.log_artifact(test_arr)

        model_trainer = ModelTrainer()
        r2_score = model_trainer.initiate_model_training(train_arr, test_arr)
        mlflow.log_metric("R2_Score", r2_score)
