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
    
    def left_join_dataframes(self, right_df, left_index=True, right_index=True):
        '''
        This functions joins on the index of the LEFT DataFrame
        '''
        joined_df = pd.merge(self.df, right_df, left_index=True, right_index=True)
        return joined_df

# Research how the left join works, the problem is likely to be the index that it is joining on

# Testing left_join_dataframes 

# def left_join_dataframes(left_df, right_df):
#     '''
#     This functions joins on the index of the LEFT DataFrame
#     '''
#     joined_df = pd.merge(left_df, right_df, left_index=True, right_index=True)
#     return joined_df


#joined_df = pd.merge(left_df, right_df, left_index=True, right_index=True)
#joined_df

# This works 
# Step 2 
#Convert 'Type' into Binary Categories
transform = DataTransform(failure_data)
type_dummies = transform.create_dummies_from_column(failure_data['Type']) 

# Join the generated 'Type' Categories DF onto the original df
failure_data = transform.left_join_dataframes(type_dummies)
failure_data.head()
#failure_data.info()
transform.left_join_dataframes() # Success 

# Try Step 1

transform = DataTransform(failure_data)
transform.convert_column_to_category(column_name='Type').info()


# Try together 
type_dummies = transform.create_dummies_from_column(failure_data['Type']) 

transform.left_join_dataframes(type_dummies) # chaining doesn't work 


########################################################################################
###### Step 1 & Step 2 work seperetely 
########################################################################################
# This works 
# Step 1 
transform = DataTransform(failure_data)
transform.convert_column_to_category(column_name='Type').info()
########################################################################################
# This works 
# Step 2 
#Convert 'Type' into Binary Categories
transform = DataTransform(failure_data)
type_dummies = transform.create_dummies_from_column(failure_data['Type']) 

type_dummies.sum()

# Join the generated 'Type' Categories DF onto the original df
failure_data = transform.left_join_dataframes(type_dummies)
failure_data.head()
failure_data.info()
####################################################################################
# Trying steps 1 & 2 together as a chain

transform = DataTransform(failure_data)

# first create dummies of the 'Type' variables
type_dummies = transform.create_dummies_from_column('Type')

transform.left_join_dataframes('type_dummies').convert_column_to_category(column_name='Type')

####################################################################################
# The export the data to continue with the df 
def export_data_as_csv(data, file_name):
    '''
    Exports data as .csv file
    '''
    data.to_csv(f"{file_name}.csv")

export_data_as_csv(failure_data, 'failure_data_after_data_transformation')

########################################################################################
###### Step 1 & Step 2 DO NOT work seperetely
########################################################################################
# TODO: Get the Step 1 & Step 2 working together 

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


# Step 2b) 
# Join the generated 'Type' Categories DF onto the original df
#transform_2.left_join_dataframes(type_dummies)
failure_data_2 = transform_2.left_join_dataframes(type_dummies)
failure_data_2.head()
failure_data_2.info()


# test 
df_1 = failure_data_1.head()
df_2 = type_dummies.head()

# TODO

joined_df = pd.merge(df_1, df_2, how = 'left', left_index=True)
joined_df

transform_test = DataTransform(df_1)
#transform_test.left_join_dataframes(df_2) # Fails

# Test the left_join_dataframe func 
def left_join_dataframes(left_df, right_df):
    joined_df = left_df.join(right_df)
    return joined_df

left_join_dataframes(df_1, df_2) 






