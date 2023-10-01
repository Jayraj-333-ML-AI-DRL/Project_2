import  os
import sys 
from src.MLproject.exception import custom_exception
from src.MLproject.logger import logging
import pandas as pd 
from dotenv import load_dotenv
import pymysql


import pickle
import numpy 

load_dotenv()

host = os.getenv('host')
user = os.getenv('user')
password= os.getenv('password')
db = os.getenv('db')

def read_sql_data():
    try:
        mydb = pymysql.connect(host=host,
                               user=user,
                               password=password,
                               db=db)
        
        
        logging.info('Connection established')
        
        
        df = pd.read_sql_query('select * from studentsperformance ', mydb)
        
        print(df.head())
        return df
    except Exception as e:
        raise custom_exception(e,sys)


def save_obj(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path,exist_ok=True)
        
        
        with open(file_path,'wb') as f:
            pickle.dump(obj,f)
            
            
    except Exception as e :
        raise custom_exception(e,sys)