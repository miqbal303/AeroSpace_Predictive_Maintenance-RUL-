import os
import sys
import numpy as np
from dataclasses import dataclass
import mlflow
from mlflow import log_params, log_metrics, log_artifact
from src.Turbo_Engine_Predict_Maintenance.logger import logging
from src.Turbo_Engine_Predict_Maintenance.exception import CustomException
from src.Turbo_Engine_Predict_Maintenance.utils import save_object, evaluate_models
from sklearn.metrics import r2_score, mean_absolute_error,mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            mlflow.start_run(nested=True)
            mlflow.set_experiment("ModelTraining")

            logging.info("Splitting training and test data inputs")
            X_train, y_train, X_test, y_test = (
                                                train_array[:, :-1],
                                                train_array[:, -1],
                                                test_array[:, :-1],
                                                test_array[:, -1]
                                            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "XGBRegressor": XGBRegressor(),
                "SupportVector Regressor": SVR(),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            params = {
                    'Random Forest': {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [3, 5, 7],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4]
                    },
                    'Gradient Boosting': {
                        'n_estimators': [50, 100, 200],
                        'learning_rate': [0.1, 0.05, 0.01],
                        'max_depth': [3, 5, 7],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4]
                    },
                    'XGBRegressor': {
                        'n_estimators': [50, 100, 200],
                        'learning_rate': [0.1, 0.05, 0.01],
                        'max_depth': [3, 5, 7],
                        'min_child_weight': [1, 3, 5]
                    },
                       'SupportVector Regressor' :{
                         'C': [0.1, 1, 5],
                         'kernel': ['linear', 'rbf',],
                         'degree': [2, 3, 4],
                         'epsilon': [0.1, 0.01, 0.2 ,0.001]
                    },
                    'AdaBoost Regressor': {
                        'n_estimators': [50, 100, 200],
                        'learning_rate': [0.1, 0.05, 0.01]
                    },
                   
                    
                     }
            
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=params)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                print("No best model found")
            logging.info("No best model found")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            mae = mean_absolute_error(y_test, predicted)
            rmse = np.sqrt(mean_squared_error(y_test, predicted))
            logging.info(f"Best found model is {best_model}, R2 Score : {r2_square}")
            print(f'Best Model Found , Model Name : {best_model} , R2 Score : {r2_square}')

            # Log parameters, metrics, and artifacts
            log_params({
                "TrainedModelPath": self.model_trainer_config.trained_model_file_path,
            })

            log_metrics({
                "R2_Score": r2_square,
                "Mean_Absolute_Error": mae,
                "RMSE": rmse
            })

            log_artifact(self.model_trainer_config.trained_model_file_path)

            return r2_square

        except Exception as e:
            #mlflow.log_exception(e)
            raise CustomException(e, sys)
        finally:
            mlflow.end_run()
