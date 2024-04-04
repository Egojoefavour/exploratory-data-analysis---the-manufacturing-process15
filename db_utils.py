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
    
df = RDSDatabaseConnector().load_data(saved_file= "data_pandas.csv")   


class DataTransform:
    '''
    converts column into required formats
    (milestone 3 task 1)
    methods:
    ------
    to_categorical()
    '''
    
    
    @staticmethod
    def to_categorical(df, col):
        '''
        converts column types to categorical data
        
        parameters:
        ---------
        df:dataframe
        col:columns of dataframe
        '''
        df[col] = df[col].astype("category")
        return df
        

    
transformer = DataTransform()    
print(transformer.to_categorical(df, col="Type"))


class DataFrameInfo:
    '''
    contain methods that generate useful information about the DataFrame.
    (milestone 3 task 2)
    methods:
    -------
    col_dtypes()
    
    df_info()
    
    dnst_cnt()
    
    amount_null()
    
    count_null()
    
    skew_df()
    
    '''
    
    
    @staticmethod
    def col_dtypes(df):
        '''
        return the data types of dataframe columns
        
        parameters:
        ----------
        df:dataframe
        '''
        return df.dtypes
    
    @staticmethod
    def df_info(df):
        '''
        returns informations about the dataframe columns, these includes mean, median, std, 
        min value, max value, count and others information of dataframe columns
        
        parameters:
        ---------
        df:dataframe
        '''
        
        infor = df.describe()
        return infor
    
    @staticmethod
    def dnst_cnt(df, col):
        '''
        returns the number of unique values  of the dataframe columns
        
        parameters:
        ---------
        df:dataframe
        col:dataframe column
        '''
        return df[col].nunique()
    
    
    @staticmethod
    def amount_null(df):
        '''
        returns the sum of the total number of null values in a dataframe column
        
        parameters:
        ---------
        df:dataframe
        '''
        return df.isnull().sum()
     
    
    @staticmethod
    def count_null(df):
        '''
        returns the percentage of dataframe column nulls sum to dataframe count
        
        parameters:
        ---------
        df:dataframe
        '''
        count_null_percentage = df.isnull().sum()* 100/len(df)
        return count_null_percentage
    
    
    @staticmethod
    def skew_df(df):
        '''
        returns the skewness of dataframe numeric columns
        parameters:
        ---------
        df:dataframe
        '''
        return df.skew(numeric_only= True)
    
info = DataFrameInfo()
print(info.col_dtypes(df))
print(info.dnst_cnt(df, col= "Type"))
print(info.df_info(df))
print(info.amount_null(df))
print(info.count_null(df))
print(info.skew_df(df))

     
    
class  DataFrameTransform:
    '''
    This class contains method that will be used to perform EDA tranformation 
    of dataframes data
    
    methods:
    ---------
    imput_null()
    
    tw_min()
    
    tran_col()
    
    drop_col()
    
    save_df()
    
    map_type()
    
    map_pro_id()
    
    
    
    '''
    
    @staticmethod
    def imput_null(df):
        '''
        this method imput the null values in  dataframe columns with the mean of the 
        dataframe column values
        (task3, step 3)
        
        parameters:
        ---------
        df:dataframe
        
        '''
        df.fillna({'Air temperature [K]':df['Air temperature [K]'].mean(),'Process temperature [K]': 
            df['Process temperature [K]'].mean(), 'Tool wear [min]': df['Tool wear [min]'].mean()}, inplace=True)
        return df
    
    @staticmethod
    def tw_min(df):
        '''
        this method transforms the dataframe column 'Tool wear [min]' to 
        minutes from seconds
        
        parameters:
        ----------
        df:dataframe
        '''
        df['Tool wear [min]'] = df['Tool wear [min]'] / 60
        return df
    
    @staticmethod
    def tran_col(df):
        ''''
        this method is used to transform the dataframe column, 'Rotational speed [rpm]'  to
        reduce its skewness
        
        parameters:
        ---------
        df:dataframe
        
        '''
        log_trans = df['Rotational speed [rpm]'].map(lambda i: np.log(i) if i > 0 else 0)
        df['Rotational speed [rpm]'] = log_trans
        return df
    
    @staticmethod
    def map_type(df):
        '''
        In the df i noticed the column 'Tool wear [min]' does not corrrelate with 
        the the product quality type H:M:L which should have this value of 'Tool wear [min]'
        5:3:2 respectively.
        This method corrects that by returning the correct product quality type for the correct
        'Tool wear [min]' values. This is done by mapping the function map_tl()  to the 'Tool wear [min]'
        column
        
        parameters:
        ---------
        df:dataframe
        'Type': Quality of the product being created 
                (L, M, or H, for low, medium and high quality products)
        'Tool wear [min]':The current minutes of wear on the tool. H, M and L product
                          manufacturing cause 5/3/2 minutes of tool wear.
        methods:
        -------
        map_tl()
        '''
        def map_tl(x):
            '''
            this method is used to iterate into the 'Tool wear [min]' column to return
            H:M:L for the 'Tool wear [min]' values 5:3:2 respectively
            
            parameters:
            ---------
            x:dataframe column 'Tool wear [min]' values
            '''
            if 5 >= x >= 3:
                return 'H'
            elif 3 >= x >= 2:
                return 'M'
            elif 2 >= x >= 0:
                return 'L'
                pass
            
        df['Type'] = df['Tool wear [min]'].map(map_tl)
        return df
        
    @staticmethod    
    def map_pro_id(df):
        '''
        This method correct the non numeric part of the 'Product ID' column which is the same as 
        the column 'Type' by replacing it with the modified df['Type']
        
        parameters
        --------
        df:dataframe
        'Product ID':Product specific serial number column
        'Type':Product quality type
        '''
        df['Product ID']= df['Type'].astype("str") + df['Product ID'].astype("str").str.slice(1)
        return df    
        
    @staticmethod
    def drop_col(df, col):
        '''
        This method is used to drop unwanted columns of the dataframe
        
        parameters:
        ---------
        df:dataframe
        col:dataframe column to be dropped
        '''
        df.drop(columns=[col], inplace=True)
        return df
    
    
    @staticmethod
    def rm_rpm_outl(df):
        '''
        This method is used to remove the outliers from the 'Rotational speed [rpm]'column
        of the dataframe
        
        parameters:
        ----------
        df:column dataframe
        '''
        df.where(df['Rotational speed [rpm]'] <= 1830).dropna()
        return df
    
    
    @staticmethod
    def rm_tq_outl(df):
        '''
        This method is used to remove the outliers from the ['Torque [Nm]' column
        of the dataframe
        
        parameters:
        ----------
        df:column dataframe
        '''
        df.where(df['Torque [Nm]'] <= 66).dropna()
        return df
    
    
    
    @staticmethod
    def save_df(df):
        '''
        This method is used to save Transformed data of the dataframe to your local
        machine
        
        parameters:
        df:dataframe to be saved
        'new_df.csv': saved dataframe in the your local machine
        '''
        df.to_csv('new_df.csv', index=False)
    
    
    
    @staticmethod
    def range_seldfcol(df):
        '''
        returns a dataframe of the min and max values of of the selected dataframe columns
        
        parameters:
        ----------
        df: dataframe
        '''
        return df[['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 
               'Torque [Nm]','Tool wear [min]']].agg(['min', 'max'])
    
    @staticmethod
    def range_sel_df_col_H(df):
        '''
        returns a dataframe of the min and max values of of the selected dataframe columns grouped by column Type 'H'
        
        '''
        return df[['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 
               'Torque [Nm]','Tool wear [min]']][df['Type'] == 'H'].agg(['min', 'max'])                                                                                            
    
    @staticmethod
    def range_sel_df_col_M(df):
        '''
        returns a dataframe of the min and max values of of the selected dataframe columns grouped by column Type 'M'
        
        '''
        return df[['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 
                'Torque [Nm]','Tool wear [min]']][df['Type'] == 'M'].agg(['min', 'max'])
    
    @staticmethod
    def range_sel_df_col_L(df):
        '''
        returns a dataframe of the min and max values of of the selected dataframe columns grouped by column Type 'L'
        
        '''
        return df[['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 
              'Torque [Nm]','Tool wear [min]']][df['Type'] == 'L'].agg(['min', 'max'])
        
    
   
transfm = DataFrameTransform()

transfm.imput_null(df)
transfm.tw_min(df)
transfm.map_type(df)
transfm.map_pro_id(df)
transfm.rm_rpm_outl(df)
transfm.rm_tq_outl(df)
transfm.save_df(df)
print(transfm.range_seldfcol(df))
print(transfm.range_sel_df_col_H(df))
print(transfm.range_sel_df_col_M(df))
print(transfm.range_sel_df_col_L(df))







class Plotter:
    '''
    This class has methods that visualizes insights from the dataframe or dataframe columns
    
    methods:
    -------
    visual_null()
    
    visual_skew()
    
    co_ma()
    
    d_liers()
    
    
    '''
    
    @staticmethod
    def visual_null(df):
        '''
        this method visualizes dataframe columns that has null values
        
        parameters:
        ---------
        df:dataframe
        '''
        plt.figure(figsize= (8, 4))
        return msno.matrix(df)
    
    @staticmethod
    def visual_skew(df, col):
        '''
        this method visualizes the skewness of a dataframe column
        
        parameters:
        ----------
        df:dataframe
        col:dataframe column to be visualise for skewness
        '''
        plt.figure(figsize= (8, 4))
        sns.histplot(df, x= df[col], kde=True)
        return sns.despine()
    
    
    @staticmethod        
    def co_ma(df):
        '''
        this method visualize the correlation matrix of the dataframe columns
        (task 6)
        parameters:
        ---------
        df:dataframe
        '''
        plt.subplots(figsize=(10, 5))
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', linewidths= .5, fmt= ".2%")
        plt.tight_layout()
        return   plt.show()
    
    @staticmethod
    def d_liers(df, col):
        '''
        this method of visualization helps in detecting if a dataframe column has an 
        outlier
        (task 5 step 1 and 3)
        
        parameters:
        ---------
        df:dataframe
        col:dataframe column to visualized for the presence of outliers
        '''
        df.boxplot(column = col,grid=False,  fontsize=15,  figsize=(15, 8)) 
        return plt.show()
    
    @staticmethod
    def q_plot(df, col):
        '''
        this method also helps in detecting outliers in dataframe columns
        
        parameters:
        ---------
        df:dataframe
        col:dataframe column to be visualized for outliers
        '''
        qqplot(df['Rotational speed [rpm]'] , scale=1 ,line='q', fit=True)
        return plt.show()
    
    
    @staticmethod
    def vis_t_w(df):
        '''
        this method of visualization helps in  displaying the number of tools operating at different tool wear values. 
        
        (milstone 4 task 1)
        
        parameters:
        ---------
        df:dataframe
        
        '''
        df.groupby('Type').boxplot(column= 'Tool wear [min]', grid=False,  fontsize=15,  figsize=(15, 8))
        return plt.show()
    
    
    @staticmethod
    def fal_rt(df):
        df[['Machine failure','TWF','HDF','PWF','OSF','RNF']].agg('sum').plot(kind='bar',
        figsize=(15, 8), ylabel= 'NUMBERS OF FAILURES', fontsize= 15)
        return plt.show()
    
     
    @staticmethod        
    def co_fal_rt(df):
        '''
        this method visualize the correlation matrix of the selected dataframe columns
        (milestone4, Task 3)
        parameters:
        ---------
        df:dataframe
        '''
        plt.subplots(figsize=(10, 5))
        sns.heatmap(df[['Air temperature [K]','Torque [Nm]','Tool wear [min]',
                  'Machine failure','TWF','HDF','PWF','OSF','RNF']].corr(numeric_only=True), 
                  annot=True, cmap='coolwarm', linewidths= .5, fmt= ".2%")
        plt.tight_layout()
        return   plt.show()


plota = Plotter()
print(plota.visual_null(df))
print(plota.visual_skew(df, col= 'Rotational speed [rpm]'))
print(plota.co_ma(df))
print(plota.d_liers(df, col= 'Rotational speed [rpm]'))
print(plota.q_plot(df, col='Rotational speed [rpm]'))
print(plota.vis_t_w(df))
print(plota.fal_rt(df))
print(plota.co_fal_rt(df))




















