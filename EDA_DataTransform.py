# Imports
import pandas as pd

# Load the Data into a data frame 
failure_data = pd.read_csv("failure_data.csv")
failure_data.head()

# TODO:
# Create a class for Data Transformations 

class DataTransform:
    def __init__(self, df):
        self.df = df

    def return_shape(self):
        return str(self.df.shape) 
    
    def return_info(self):
        return self.df.info()
    
    def return_first_row(self):
        return self.df.iloc[0]
    
    def unique_obervations(self, column):
        return column.unique()
    
    def convert_column_to_category(self, column):
        '''
        converts the dtype of column to 'category'
        '''
        column = pd.Categorical(column)
    
    def create_dummies_from_column(self, column):
        dummies_df = pd.get_dummies(column, dtype= int)
        return dummies_df 
    
    def left_join_dataframes(self, right_df):
        joined_df = self.df.join(right_df)
        return joined_df

# Create an instance 
transform = DataTransform(failure_data)

# Testing the Class Methods 

# Testing return methods
transform.return_shape() # Success
transform.return_info() # Success
transform.return_first_row() # Success 

# Testing column transformations
transform.unique_obervations(failure_data['Type']) # Success
transform.create_dummies_from_column(failure_data['Type']) # Success

# Testing join 
type_dummies = transform.create_dummies_from_column(failure_data['Type']) # Success 
transform.left_join_dataframes(type_dummies) # Success 
