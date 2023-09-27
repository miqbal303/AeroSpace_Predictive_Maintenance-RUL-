import mlflow
import mlflow.sklearn
import os
import pandas as pd
from dataclasses import dataclass
from pymongo import MongoClient
import logging  # Import the logging module
from src.Turbo_Engine_Predict_Maintenance.exception import CustomException
from src.config import MONGODB_URI, DB_NAME, COLLECTION_NAMES
from src.Turbo_Engine_Predict_Maintenance.utils import connect_to_mongodb, fetch_data_from_mongodb, remove_id_field

# Configure the logging module
logging.basicConfig(filename='data_ingestion.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class DataIngestionConfig:
    rul_data_path: str = os.path.join('artifacts', 'Rul.csv')
    train_data_path: str = os.path.join('artifacts', 'Train.csv')
    test_data_path: str = os.path.join('artifacts', 'Test.csv')

class DataIngestion:
    def __init__(self, uri, db_name, collection_names):
        self.uri = uri
        self.db_name = db_name
        self.collection_names = collection_names
        self.ingestion_config = DataIngestionConfig()

    def connect_to_mongodb(self):
        try:
            mlflow.log_param("MongoDB_URI", self.uri)
            mlflow.log_param("MongoDB_DBName", self.db_name)

            client = MongoClient(self.uri)
            print(f"Selected database: {self.db_name}")
            logging.info("Selected database: %s", self.db_name)  # Log this information
            self.db = client[self.db_name]
            self.collections = {name: self.db[name] for name in self.collection_names}
            print("Connected to MongoDB!")
            logging.info("Succesfully Connected to MongoDB!")
        except Exception as e:
            mlflow.log_exception(e)
            logging.error("Exception while connecting to MongoDB: %s", str(e))  # Log the error
            raise e

    def fetch_data_from_mongodb(self, collection_name):
        try:
            collection = self.collections[collection_name]
            all_documents = list(collection.find())

            # Remove the '_id' field from each document
            for document in all_documents:
                remove_id_field(document)

            df = pd.DataFrame(all_documents)
            return df
        except Exception as e:
            mlflow.log_exception(e)
            logging.error("Exception while fetching data from MongoDB: %s", str(e))  # Log the error
            raise e

    def save_data_to_csv(self, df, file_path):
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            df.to_csv(file_path, index=False)
            mlflow.log_artifact(file_path)
            print(f'Data saved to {file_path}')
            logging.info("Data saved to %s", file_path)  # Log this information
        except Exception as e:
            mlflow.log_exception(e)
            logging.error("Exception while saving data to CSV: %s", str(e))  # Log the error
            raise e

    def initiate_data_ingestion(self):
        print('Data Ingestion start')
        try:
            # Connect to MongoDB
            self.connect_to_mongodb()

            # Fetch data from MongoDB and convert to DataFrame
            train_df = self.fetch_data_from_mongodb("Train")
            test_df = self.fetch_data_from_mongodb("Test")
            rul_df = self.fetch_data_from_mongodb("Rul")

            # Save the train and test and Rul data to CSV files
            self.save_data_to_csv(train_df, self.ingestion_config.train_data_path)
            self.save_data_to_csv(test_df, self.ingestion_config.test_data_path)
            self.save_data_to_csv(rul_df, self.ingestion_config.rul_data_path)

            print('Ingestion of Data is completed')
            logging.info("Data Ingestion completed")  # Log this information
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                self.ingestion_config.rul_data_path,
            )

        except Exception as e:
            mlflow.log_exception(e)
            logging.error("Exception occurred at data ingestion stage: %s", str(e))  # Log the error
            print('Exception occurred at data ingestion stage')
            raise e

if __name__ == '__main__':
    with mlflow.start_run():
        mlflow.set_experiment("DataIngestion")
        obj = DataIngestion(MONGODB_URI, DB_NAME, COLLECTION_NAMES)
        train_data_path, test_data_path, rul_data_path = obj.initiate_data_ingestion()
