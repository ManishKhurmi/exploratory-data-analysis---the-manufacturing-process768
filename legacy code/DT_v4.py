# Imports
import pandas as pd

# Create a class for Data Transformations 
class DataTransform:
    def __init__(self, df):
        self.df = df
        self.info = df.info()

    def return_shape(self):
        return str(self.df.shape) 
    
    def return_info(self):
        return self.df.info()
    
    def return_first_row(self):
         return self.df.iloc[0]

    def unique_observations(self, column_name):
        return self.df[column_name].unique()
    
    def convert_column_to_category(self, column_name):
        '''
        converts the dtype of column to 'category'
        '''
        self.df[column_name] = pd.Categorical(self.df[column_name])
        return self.df
    
    def create_dummies_from_column(self, column_name):
        dummies_df = pd.get_dummies(self.df[column_name], dtype=int)
        return dummies_df 

    def concat_dataframes(self, new_df, left_index=True, right_index=True):
        '''
        This functions joins on the index of the LEFT DataFrame
        '''
        joined_df = pd.concat([self.df, new_df], axis = 1)
        return joined_df
    
if __name__=='__main__': 

    # Load the Data into a data frame 
    failure_data = pd.read_csv("failure_data.csv")

    transform = DataTransform(failure_data)

    # info methods
    transform.return_shape()
    transform.return_info()

    #Data Transform Methods
    transform.return_first_row()
    transform.unique_observations('UDI')
    failure_data_converted = transform.convert_column_to_category('Type')
    print(failure_data_converted.info)
    print(failure_data_converted)

    transform = DataTransform(failure_data)

    # Test 
    transform.return_shape()
    transform.return_info()
    transform.return_first_row()
    transform.unique_observations('UDI')
    transform.convert_column_to_category('Type').info()
    transform.create_dummies_from_column('Type')

    # Test 
    df1 = transform.create_dummies_from_column('Type')
    print(df1)
    joined_df = transform.concat_dataframes(df1)
    print(joined_df)
    joined_df.info()


# Data transformations 
 # Type -> categories 
 # create dummies of the Type data 
 # treat for null data 
