import os
import sys
from pathlib import Path
import logging

# Configure logging to display INFO level messages
logging.basicConfig(level=logging.INFO)

# Define the project name
project_name = 'MLproject'

# List of files to be created
list_of_files = [
    f"src/{project_name}/pipelines/__init.py",
    f"src/{project_name}/pipelines/training_pipeline.py",
    f"src/{project_name}/pipelines/prediction_pipeline.py",
    f"src/{project_name}/components/__init.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/data_transformation.py",
    f"src/{project_name}/components/model_trainer.py",
    f"src/{project_name}/components/model_monitoring.py",
    f"src/{project_name}/exception.py",
    f"src/{project_name}/logger.py",
    f"src/{project_name}/utils.py",
    "app.py",
    "Dockerfile",
    "requirements.txt",
    "setup.py"
     ]

# Loop through the list of files
for filepath in list_of_files:
    # Convert the file path to a Path object
    filepath = Path(filepath)
    # Extract directory and filename from the file path
    filedir, filename = os.path.split(filepath)
    
    # Check if the directory is not empty
    if filedir != "":
        # Create the directory if it does not exist
        os.makedirs(filedir, exist_ok=True)
        # Log the creation of the directory
        logging.info(f"Creating Directory: {filedir} for the file {filename}")
    
    # Check if the file does not exist or is empty
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        # Create an empty file
        with open(filepath, "w") as f:
            pass
        # Log the creation of the empty file
        logging.info(f"Creating empty file: {filepath}")
    else:
        # Log if the file already exists
        logging.info(f"{filename} already exists")
