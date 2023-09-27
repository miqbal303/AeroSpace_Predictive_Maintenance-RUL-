import sys
import traceback
import mlflow

def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename

    error_message = "Error occurred in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )

    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message

    def log_exception_to_mlflow(self):
        """
        Log exceptions to MLflow
        """
        mlflow.log_params({"Exception": str(self)})
        mlflow.log_params({"ExceptionStackTrace": traceback.format_exc()})

#if __name__ == '__main__':
#    try:
#        # Your code here...
#    except Exception as e:
#        custom_exception = CustomException(e, sys)
#        custom_exception.log_exception_to_mlflow()
