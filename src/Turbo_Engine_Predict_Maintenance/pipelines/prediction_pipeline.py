import sys
import os
import numpy as np
import pandas as pd
import mlflow
from mlflow import log_params, log_metrics, log_artifact
from src.Turbo_Engine_Predict_Maintenance.logger import logging
from src.Turbo_Engine_Predict_Maintenance.exception import CustomException
from src.Turbo_Engine_Predict_Maintenance.utils import load_object, save_object
import traceback

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            mlflow.start_run()
            mlflow.set_experiment("PredictionPipeline")

            # Save model and preprocessor as artifacts
            

            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            #save_object(model, file_path=model_path)
            #save_object(preprocessor, file_path=preprocessor_path)

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)

            # Log parameters, metrics, and artifacts
            log_params({
                "ModelPath": model_path,
                "PreprocessorPath": preprocessor_path
            })

            # You can log additional metrics here if needed
            # log_metrics({
            #     "MetricName": metric_value
            # })

            # You can log additional artifacts here if needed
            # log_artifact(artifact_path)

            return preds

        except Exception as e:
            # Truncate the traceback message if it's too long
            exception_message = str(e)
            exception_stack_trace = traceback.format_exc()[:500]  # Truncate to 500 characters
            mlflow.log_params({"exception_message": exception_message})
            mlflow.log_params({"exception_stack_trace": exception_stack_trace})
            raise CustomException(e, sys)
        finally:
            mlflow.end_run()


class CustomData:
    def __init__(self,
                 engine_number: float,
                 time_cycles: float,
                 sensor_measurement2: float,
                 sensor_measurement3: float,
                 sensor_measurement4: float,
                 sensor_measurement7: float,
                 sensor_measurement8: float,
                 sensor_measurement9: float, 
                 sensor_measurement11: float,
                 sensor_measurement12: float,
                 sensor_measurement13: float,
                 #sensor_measurement15: float,
                 #sensor_measurement17: float,
                 #sensor_measurement20: float,
                 #sensor_measurement21: float
                 ):
        
        self.engine_number = engine_number
        self.time_cycles = time_cycles
        self.sensor_measurement2 = sensor_measurement2
        self.sensor_measurement3 = sensor_measurement3
        self.sensor_measurement4 = sensor_measurement4
        self.sensor_measurement7 = sensor_measurement7
        self.sensor_measurement8 = sensor_measurement8
        self.sensor_measurement9 = sensor_measurement9
        self.sensor_measurement11 = sensor_measurement11
        self.sensor_measurement12 = sensor_measurement12
        self.sensor_measurement13 = sensor_measurement13
        #self.sensor_measurement15 = sensor_measurement15
        #self.sensor_measurement17 = sensor_measurement17
        #self.sensor_measurement20 = sensor_measurement20
        #self.sensor_measurement21 = sensor_measurement21
    
    def get_data_as_data_frame(self):
        try:
            mlflow.start_run(nested=True)
            mlflow.set_experiment("CustomData")
            
            custom_data_input_dict = {
                     "engine_number" : [self.engine_number],
                     "time_cycles" : [self.time_cycles],
                     "sensor_measurement2"  : [self.sensor_measurement2], 
                     "sensor_measurement3"  : [self.sensor_measurement3], 
                     "sensor_measurement4"  : [self.sensor_measurement4], 
                     "sensor_measurement7"  : [self.sensor_measurement7], 
                     "sensor_measurement8"  : [self.sensor_measurement8], 
                     "sensor_measurement9"  : [self.sensor_measurement9], 
                     "sensor_measurement11" : [self.sensor_measurement11], 
                     "sensor_measurement12" : [self.sensor_measurement12], 
                     "sensor_measurement13" : [self.sensor_measurement13], 
                     #"sensor_measurement15" : [self.sensor_measurement15], 
                     #"sensor_measurement17" : [self.sensor_measurement17], 
                     #"sensor_measurement20" : [self.sensor_measurement20], 
                     #"sensor_measurement21" : [self.sensor_measurement21], 
                }

            df = pd.DataFrame(custom_data_input_dict)

            # Log parameters and artifacts
            log_params({
                "EngineNumber": self.engine_number,
                "TimeCycles": self.time_cycles,
            })

            # You can log additional artifacts here if needed
            # log_artifact(artifact_path)

            return df

        except Exception as e:
            # Truncate the traceback message if it's too long
            exception_message = str(e)
            exception_stack_trace = traceback.format_exc()[:500]  # Truncate to 500 characters
            mlflow.log_params({"exception_message": exception_message})
            mlflow.log_params({"exception_stack_trace": exception_stack_trace})
            raise CustomException(e, sys)
        finally:
            mlflow.end_run()
