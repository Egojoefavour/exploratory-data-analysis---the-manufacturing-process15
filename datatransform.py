import pandas as pd
import numpy as np




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
        

if __name__ == "__main__":    
   transformer = DataTransform()    
   print(transformer.to_categorical(df, col="Type"))