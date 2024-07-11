# Imports
import pandas as pd

# Load the Data into a data frame 
failure_data = pd.read_csv("failure_data.csv")
failure_data.head()

# Shape of DF 
failure_data.shape # 10,000 observations and 15 variables 

failure_data.info() 
# The data types are what I would expect 

# Task 1: Convert columns into the correct format 

# Check 1 row to view all variables 
failure_data.iloc[0]

# My Data transformations:
    # might consider converting the "Type" variable into a string object type 
        # Convert "Type" into categorical variables e.g. L = 1


failure_data['Type'].unique() # L, M & H for the categories 

# convert "Type" variable into strings 
 # Note that standardising the Letters is not needed as there are only the three unique Letter types e.g. only "L" present rather than "l" and "L" etc
failure_data['Type'] = failure_data['Type'].astype('string') # by default sets the length to the max len it encounters 

# check our transformation worked 
failure_data['Type'].dtype # Proves the conversion 

# Now convert into "Type" into binary categories 
type_categories = pd.get_dummies(failure_data['Type'], dtype= int)
type_categories.info()
type_categories.head()

failure_data_with_type_dummies = failure_data.join(type_categories)
failure_data_with_type_dummies.info()
failure_data_with_type_dummies.head()

###################################################################################################
# TODO: convert the above into functions and create class

#failure_data.shape 
def shape(df):
    return str(df.shape) 
# Test
#shape(failure_data)

#failure_data.info() 
def info(df):
    return df.info()
# Test 
#info(failure_data)

#failure_data['Type'].unique()
def unique_obervations(column):
    return column.unique()
# Test 
#unique_obervations(failure_data['Type'])


#failure_data.iloc[0]
def first_row(df):
    return df.iloc[0]
#Test 
#first_row(failure_data)

###############

failure_data['Type'] = failure_data['Type'].astype('string') # by default sets the length to the max len it encounters 

# Doesn't work 
#def convert_column_to_string(col): 
#    return col.astype('string') 

# Works but clunky 
#def convert_column_to_string(col):
#    return col.astype('string')
#failure_data['Type'] = convert_column_to_string(failure_data('Type'))

def convert_col_to_string(df, col):
    failure_data['Type'] = failure_data['Type'].astype('string')

##############
# # How can I call this local variable outside the function? 

# # I've found the following solution but it might cause problems when creating a class due to needing the column_name:
    
# def convert_column_to_string(df, column_name):
#     df[column_name] = df[column_name].astype('string')

# # Test
# convert_column_to_string(failure_data, 'Type')
# failure_data.info()

# # Whats even better is to convert it to 'category type' rather than string
# failure_data['Type'] = pd.Categorical(failure_data['Type'])
# failure_data.info()

def convert_column_to_category(column):
    '''
    converts the dtype of column to 'category'
    '''
    column = pd.Categorical(column)

convert_column_to_category(failure_data['Type'])
failure_data.info()



###########

def create_dummies_from_column(col):
    dummies_df = pd.get_dummies(col, dtype= int)
    return dummies_df 

# Test 
create_dummies_from_column(failure_data['Type']) 


###########
# Successfull 
  
# def left_join_dummies_to_df(original_df, dummy_var):
#     original_df_with_dummies = original_df.join(dummy_var)
#     return original_df_with_dummies

# # Test
# type_dummy = create_dummies_from_column(failure_data['Type'])
# left_join_dummies_to_df(failure_data, type_dummy)

# failure_data_with_type_dummies = left_join_dummies_to_df(failure_data, type_dummy)
# failure_data_with_type_dummies.info()

# change syntax 
# def left_join_to_df(df_1, df_2):
#     df_with_join = df_1.join(df_2)
#     return df_with_join

def left_join_dataframes(df, right_df):
    joined_df = df.join(right_df)
    return joined_df
# Test 
#left_join_to_df(failure_data,create_dummies_from_column(failure_data['Type']))
# Success


###################################################################################################

# Are the UDI & Product ID unique? 

###################################################################################################
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








    
#TODO: use Jared's advice in Approach 2 to create the methods for this class 



    


    # Methods 


        # Convert Type into categories type 
        # Convert Type into Categories 


