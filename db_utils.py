import yaml
import psycopg2
import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from scipy.stats import skew
from statsmodels.graphics.gofplots import qqplot
import numpy as np



class RDSDatabaseConnector:
    '''
    In this project, Exploratory Data Analysis - The Manufacturing Process, this class, 
    RDSDatabaseConnector contains the methods that is used to extract data from the RDS database
    
    
    parameters:
    -----------
    dict_data: dictionary
    dictionary of credentials which will be used to extract the data from RDS database
    
    methods:
    --------
    load_yaml_to_dict()
    
    connect_db()
    
    extract_data()
    
    save_data()
    
    load_data(saved_file)
    
        
    '''
    def __init__(self):
        self.dict = self.load_yaml_to_dict()
        self.engine = self.connect_db()
        self.data_pandas = self.extract_data()
        
        
        
    def load_yaml_to_dict(self):
        '''
        creates a dictionary of credientials that will be used to extract the data from the
        RDS database. loads the dictionary from credentiials.yaml file
        
        '''
        with open('credentials.yaml', 'r') as file:
           self.dict_data = yaml.safe_load(file)
           return self.dict_data
       
       
    def connect_db(self): 
        '''
        initializes a SQLAlchemy engine from the credentials  and together with pandas 
        library allows you to extracts the data from the RDS database
        
        '''     
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
        '''
        extracts data stored in a table  called failure_data in the RDS database and 
        return it as a pandas dataframe
        '''
        self.data_pandas = pd.read_sql_table("failure_data", self.engine) 
        return self.data_pandas
     
        
    def save_data(self):
        '''
        saves data to a .csv format in your local machine
        '''
        self.extract_data()
        self.data_pandas.to_csv("data_pandas.csv", index=False)
        
        
    def load_data(self, saved_file):
        '''
        load data from your local machine into a pandas dataframe
        '''
        self.saved_file = "data_pandas.csv"
        self.load_csv = pd.read_csv("data_pandas.csv")
        return self.load_csv
    
if __name__ == "__main__":    
    df = RDSDatabaseConnector().load_data(saved_file= "data_pandas.csv")   























