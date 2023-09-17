import logging 
import os,sys
from datetime import datetime


LOG_FILE = f"{datetime.now().strftime('%m_%d_%y_%H_%m_%S')}.log"
log_path = os.path.join(os.getcwd(),"logging",LOG_FILE)
os.makedirs(log_path,exist_ok=True)

LOG_FILE_PATH = os.path.join(log_path,LOG_FILE)


logging.basicConfig(
    filename=LOG_FILE_PATH,
    format= '[%(asctime)s ] %(levelname)-8s %(name)-15s %(message)s',
    level=logging.INFO
    )
 