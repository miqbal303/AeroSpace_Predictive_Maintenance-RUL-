import os
import sys
from pathlib import Path

while True:
    project_name = input("Enter project name")
    if project_name != "":
        break


# List of file paths to be created
list_of_files = [
    "github/workflow/.gitkeep",
    #f"notebook/EDA.ipynb",
    #f"notebook/Model_Training.ipynb",
    #f"notebook/datasets/sample.csv",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/exception.py",
    f"src/{project_name}/logger.py",
    f"src/{project_name}/utils.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/data_transformation.py",
    f"src/{project_name}/components/model_trainer.py",
    f"src/{project_name}/pipelines/__init__.py",
    f"src/{project_name}/pipelines/prediction_pipeline.py",
    f"src/{project_name}/pipelines/training_pipeline.py",
    "setup.py",
    "app.py",
    "Dockerfile",
    "requirements.txt"
]

# Loop through the list of file paths and create necessary directories and files
for file_path in list_of_files:
    file_path = Path(file_path)  # Convert the file_path to a Path object for easier manipulation
    file_dir = file_path.parent  # Get the parent directory of the file
    file_name = file_path.name  # Get the name of the file or the last component of the path

    if file_dir != "":
        os.makedirs(file_dir, exist_ok=True)  # Create the directory if it doesn't exist (exist_ok=True avoids raising an error if the directory already exists)

    # Check if the file doesn't exist or if it's an empty file
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        with open(file_path, 'w') as f:
            pass  # If the file doesn't exist or is empty, create an empty file by opening and closing it
    else:
        print(f"File '{file_name}' already exists.")  # If the file exists and is not empty, print a message

print("Files created")  # Print a message to indicate that the files have been created