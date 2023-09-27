import logging
import os
from datetime import datetime
import mlflow

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

def log_exception_to_mlflow(exception):
    """
    Log exceptions to MLflow
    """
    mlflow.log_params({"Exception": str(exception)})
    mlflow.log_params({"ExceptionStackTrace": traceback.format_exc()})
    mlflow.log_artifact(LOG_FILE_PATH)

#if __name__ == '__main__':
#    try:
#        # Your code here...
#    except Exception as e:
#        logging.error("An exception occurred: %s", str(e))
#        log_exception_to_mlflow(e)
