# Imports
import pandas as pd

# Load the Data into a data frame 
failure_data = pd.read_csv("failure_data.csv")
#failure_data.head()

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
    
    def convert_column_to_category(self, column_name):
        '''
        converts the dtype of column to 'category'
        '''
        self.df[column_name] = pd.Categorical(self.df[column_name])
        return self.df
    
    def create_dummies_from_column(self, column):
        dummies_df = pd.get_dummies(column, dtype= int)
        return dummies_df 
    
    def left_join_dataframes(self, right_df):
        joined_df = self.df.join(right_df)
        return joined_df

transform_1 = DataTransform(failure_data)

# Step 1) Convert the dtype for 'Type' into categories 
failure_data_1 = transform_1.convert_column_to_category(column_name='Type')
# check using .info()
failure_data_1.info()

# Step 2) Create dummies variables for 'Type' and left join onto original df
# Step 2a) Convert 'Type' into Dummy (Binary) Categories
# Create an instance using the updated data failure_data_1
transform_2 = DataTransform(failure_data_1)
type_dummies = transform_2.create_dummies_from_column(failure_data_1['Type'])
failure_data_1
type_dummies 

# Step 2b) 
# Join the generated 'Type' Categories DF onto the original df
#transform_2.left_join_dataframes(type_dummies)
failure_data_2 = transform_2.left_join_dataframes(type_dummies)
failure_data_2.head()
failure_data_2.info()
#transform_2.left_join_dataframes()

########################################################################################
# Step 1
transform = DataTransform(failure_data)
transform.convert_column_to_category(column_name='Type')

type_dummies = transform.create_dummies_from_column(failure_data['Type']) 
transform.left_join_dataframes(type_dummies)

########################################################################################
# This works 
# Step 2 
#Convert 'Type' into Binary Categories
type_dummies = transform.create_dummies_from_column(failure_data['Type']) 

# Join the generated 'Type' Categories DF onto the original df
failure_data = transform.left_join_dataframes(type_dummies)
failure_data.head()
failure_data.info()

