# Importing necessary modules
from src.MLproject.logger import logging
import os
import sys

# Function to get error message details
def error_msg_details(error, error_details: sys):
    # Extracting file name, line number, and error message
    _, _, exc_tb = error_details.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_msg = "Error occurred in Python script [{0}] at line number [{1}] with error message: [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error))
    return error_msg

# Custom exception class
class custom_exception(Exception):
    def __init__(self, error_message, error_details: sys):
        super().__init__(error_message)
        # Generating detailed error message
        self.error_message = error_msg_details(error_message, error_details)
        
    def __str__(self):
        return self.error_message
