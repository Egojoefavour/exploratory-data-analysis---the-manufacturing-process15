import yaml
import psycopg2
import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt


class RDSDatabaseConnector:
    def __init__(self):
        self.dict = self.load_yaml_to_dict()
        self.engine = self.connect_db()
        self.data_pandas = self.extract_data()
        
        
        
    def load_yaml_to_dict(self):
        with open('credentials.yaml', 'r') as file:
           self.dict_data = yaml.safe_load(file)
           return self.dict_data
       
       
    def connect_db(self):       
        cred = self.load_yaml_to_dict()
        DATABASE_TYPE = "postgresql"
        USER = cred["RDS_USER"]
        HOST = cred["RDS_HOST"]
        PORT = cred["RDS_PORT"]
        DATABASE = cred["RDS_DATABASE"]
        PASSWORD = cred["RDS_PASSWORD"]
        self.engine = create_engine(f"{DATABASE_TYPE}+{'psycopg2'}://{USER}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}")
        
        return self.engine
        
    
    
    def extract_data(self):
        self.data_pandas = pd.read_sql_table("failure_data", self.engine) 
        return self.data_pandas
     
        
    def save_data(self):
        self.extract_data()
        self.data_pandas.to_csv("data_pandas.csv", index=False)
        
        
    def load_data(self, saved_file):
        self.saved_file = "data_pandas.csv"
        self.load_csv = pd.read_csv("data_pandas.csv")
        return self.load_csv
    
df = RDSDatabaseConnector().load_data(saved_file= "data_pandas.csv")   


class DataTransform:
    
    @staticmethod
    def to_categorical(df, col):
        df[col] = df[col].astype("category")
        return df
 

    @staticmethod
    def TW_timedelta(df, col):
        df[col] = pd.to_timedelta(df[col], unit='s')
        return df
    
DataTransform().to_categorical(df, col="Type")
DataTransform().TW_timedelta(df,  col="Tool wear[min]")

class DataFrameInfo:
    
    
    @staticmethod
    def col_dtypes(df):
        return df.dtypes
    
    @staticmethod
    def df_info(df):
        return df.describe()
    
    @staticmethod
    def dnst_cnt(df, col):
        return df[col].nunique()
     
    
    @staticmethod
    def count_null(df):
        count_null_percentage = df.isnull().sum()* 100/len(df)
        return count_null_percentage

df = DataFrameInfo().col_dtypes()
df = DataFrameInfo().df_info()
df = DataFrameInfo().dnst_cnt(col= "Type")
df = DataFrameInfo().count_null()

     
#class Plotter:
    
    
    
    
#class  DataFrameTransform: 
     