# Create the class 

import pandas as pd

# Set display options 
pd.set_option('display.max_columns', None)

# Load the data 
failure_data  = pd.read_csv('failure_data_after_data_transformation.csv')

class DataFrameInfo:
    def __init__(self, df):
        self.df = df 
    
    def print_df(self):
        print(self.df)
    
    def print_head(self):
        print(self.df.head())

    def data_type(self):
        return self.df.dtypes
    
    def describe_statistics(self):
        return self.df.describe().loc[['mean', 'std', '50%']]
    
    def unique_value_count(self, column_names):
        return self.df[column_names].nunique()
    
    def percentage_of_null(self):
        percentage_of_null = self.df.isnull().sum() / len(self.df) * 100  
        return percentage_of_null
    
    def are_all_observations_unique(self, column_name):
        print(f'The {column_name} column contains only unique rows: {len(self.df) == self.df[column_name].nunique()}')


information = DataFrameInfo(failure_data)

# Successful:
information.print_df() 
information.print_head()
information.data_type()
information.describe_statistics()
information.unique_value_count('UDI')
information.percentage_of_null()
information.are_all_observations_unique('H')
