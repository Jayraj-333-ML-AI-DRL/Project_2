from src.MLproject.logger import logging
from src.MLproject.exception import custom_exception 
import sys



if __name__ == "__main__":
    logging.info ("The executioin has started")
    
    
    try:
        a = 1/0
    except Exception as e:
        logging.info ("Custom Exception ")
        raise custom_exception(e,sys)