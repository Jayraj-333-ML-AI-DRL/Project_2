import logging 
import os
import sys
from datetime import datetime

# Define the log file name using the current date and time
LOG_FILE = f"{datetime.now().strftime('%m_%d_%y_%H_%M_%S')}.log"

# Create the path for the log file inside a 'logging' directory in the current working directory
log_path = os.path.join(os.getcwd(), "logging", LOG_FILE)

# Ensure that the directory structure exists or create it if not
os.makedirs(log_path, exist_ok=True)

# Construct the full path to the log file
LOG_FILE_PATH = os.path.join(log_path, LOG_FILE)

# Configure the logging module to write logs to the specified file
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format='[%(asctime)s ] %(levelname)-8s %(name)-15s %(message)s',
    level=logging.INFO
)
