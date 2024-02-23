import yaml
import psycopg2
import pandas as pd
from sqlalchemy import create_engine

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
        
        
    def load_data(self):
        self.save_data()
        self.load_csv = pd.read_csv("data_pandas.csv")
        return self.load_csv
    
    
love = RDSDatabaseConnector()
love.load_data()
df = love.load_data()
print(df.head(5))



 

