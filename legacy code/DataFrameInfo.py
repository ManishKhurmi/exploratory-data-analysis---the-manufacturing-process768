import pandas as pd

# Set display options 
pd.set_option('display.max_columns', None)

# Load the data 
failure_data  = pd.read_csv('failure_data_after_data_transformation.csv')
failure_data.head()

# Describe all columns in the DataFrame to check their data types
#failure_data.describe()
failure_data.dtypes # change `Type` to Categories 

# Extract statistical values: median, standard deviation and mean from the columns and the DataFrame
failure_data.describe().loc[['mean', 'std', '50%']]

# Count distinct values in categorical columns
failure_data[['Machine failure','TWF', 'HDF', 'PWF', 'OSF', 'RNF', 'H', 'L', 'M']].nunique() # All 2 signifying Binary Booleans 


# Print out the shape of the DataFrame
print(failure_data.shape)

# Generate a count/percentage count of NULL values in each column
print('Percentage of data that is NULL')
failure_data.isnull().sum() / len(failure_data) * 100 

#failure_data.isna().sum() , redundant 


#####################################################################################
# Any other methods you may find useful

# check that that each observation in the 'UDI' variable is unique 
print(f'The UDI column contains only unique rows: {len(failure_data) == failure_data['UDI'].nunique()}')

#####################################################################################
# Create the class 

import pandas as pd

# Set display options 
pd.set_option('display.max_columns', None)

# Load the data 
failure_data  = pd.read_csv('failure_data_after_data_transformation.csv')

class DataFrameInfo:
    def print_df(self):
        print(self.df)
    
    def print_head(self):
        print(self.df.head())
    
    def __init__(self, df):
        self.df = df 

    def data_type(self):
        return self.df.dtype()
    
    def describe_statistics(self):
        return self.df.describe().loc[['mean', 'std', '50%']]
    
    def unique_value_count(self, column_names):
        return self.df[column_names].nunique()
    
    def percentage_of_null(self, column_name):
        percentage_of_null = self.df.isnull().sum() / len(self.df) * 100  
        return percentage_of_null
    
    def are_all_observations_unique(self, column_name):
        print(f'The {column_name} column contains only unique rows: {len(self.df) == self.df[column_name].nunique()}')


information = DataFrameInfo(failure_data)

# Successful:
information.print_df() 
information.print_head()
    

    
