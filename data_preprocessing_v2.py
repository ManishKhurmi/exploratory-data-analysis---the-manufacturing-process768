# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import itertools
from scipy.stats import normaltest, pointbiserialr, pearsonr, chi2_contingency, yeojohnson
from scipy import stats
from statsmodels.graphics.gofplots import qqplot
import statsmodels.formula.api as smf
from IPython.display import display
from sklearn.preprocessing import StandardScaler
import DataFrameInfo_class as info
from itertools import permutations
import contextlib
import io

class DataFrameInfo:
    def __init__(self, df):
        self.df = df 
    
    def return_shape(self):
        return str(self.df.shape) 
    
    def return_info(self):
        return self.df.info()
    
    def return_first_row(self):
         return self.df.iloc[0]

    def data_type(self):
        print(self.df.dtypes)
    
    def describe_statistics(self, columns):
        # return self.df.describe().loc[['mean', 'std', '50%']]
        return self.df[columns].describe()
    
    def unique_value_count(self, column_names):
        return self.df[column_names].nunique()
    
    def percentage_of_null(self):
        percentage_of_null = self.df.isnull().sum() / len(self.df) * 100  
        return percentage_of_null
    
    def are_all_observations_unique(self, column_name):
        print(f'The {column_name} column contains only unique rows: {len(self.df) == self.df[column_name].nunique()}')
    
    def normal_test(self, column_name):
        stat, p = normaltest(self.df[column_name], nan_policy = 'omit')
        print('Statistics=%.3f, p=%.3f' % (stat, p))

    def skew_test(self, column_name):
        return self.df[column_name].skew()

    def print_mean(self, column_name):
        print(f'The mean of {column_name} is {self.df[column_name].mean()}')
    
    def print_median(self, column_name):
        print(f'The median of {column_name} is {self.df[column_name].median()}')

    def column_names(self):
        return self.df.columns

    def continous_variables(self):
        continous_variables = []
        for i in self.df.columns:
            if self.df[i].nunique() > 2:
                continous_variables.append(i)
        return continous_variables
    
    def z_score_info(self, z_scores):
        # Z-score Threshold 
        threshold_2 = 2 
        threshold_3 = 3
        # z_scores = udi_process_temp_df_z['z_scores']

        outliers_2 = (np.abs(z_scores) > threshold_2).sum() 
        outliers_3 = (np.abs(z_scores) > threshold_3).sum()

        percentage_outliers_thereshold_2 = round(outliers_2/len(z_scores) * 100, 2)
        percentage_outliers_thereshold_3 = round(outliers_3/len(z_scores) * 100, 2)

        print(f"Number of observations with outliers based on z-score threshold ±2: {outliers_2}")
        print(f"Percentage of observations with outliers based on z-score threshold ±2: {percentage_outliers_thereshold_2}")
        print("\n")
        print(f"Number of observations with outliers based on z-score threshold ±3: {outliers_3}")
        print(f"Percentage of observations with outliers based on z-score threshold ±3: {percentage_outliers_thereshold_3}")


class DataTransform:
    def __init__(self, df):
        self.df = df

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
    
    def impute_column(self, column_name, method='mean'):
        # Count the number of NULL values before filling
        null_count_before = self.df[column_name].isna().sum()

        print(f'Number of NULL values in {column_name} before imputation: {null_count_before}')
        # Choose the imputation method
        if method == 'mean':
            impute_value = self.df[column_name].mean()
        elif method == 'median':
            impute_value = self.df[column_name].median()
        elif method == 'mode':
            impute_value = self.df[column_name].mode()[0]  # Mode might return multiple values, take the first one
        else:
            raise ValueError("Method must be 'mean', 'median', or 'mode'")
        
        # Impute the missing values
        self.df[column_name] = self.df[column_name].fillna(impute_value)
        
        # Count the number of NULL values after filling (should be 0)
        null_count_after = self.df[column_name].isna().sum()
        print(f'Number of NULL values in {column_name} after imputation: {null_count_after}')
        
        return self.df[column_name]

    def yeojohnson(self, column_name):
        yeojohnson_var = self.df[column_name]
        yeojohnson_var, _ = stats.yeojohnson(yeojohnson_var) # The '_' ignores the second parameter, in this case it is the lambda parameter 
        yeojohnson_var = pd.Series(yeojohnson_var)
        return yeojohnson_var
    
    def z_score(self, column): # takes in a column and creates z scores, 
        x = self.df[column] 
        mean= np.mean(x)
        standard_deviation = np.std(x)
        z_scores = (x - mean) / standard_deviation
        return z_scores
    
    def outliers_via_z_score_df(self, column, z_threshold = [[2,3]]):

        '''
        returns a df of outliers based on the z_scores of a selected variable
        '''
        # create z scores
        x = self.df[column] 
        mean= np.mean(x)
        standard_deviation = np.std(x)
        z_scores = (x - mean) / standard_deviation

        # calculate outliers 
        outliers = np.abs(z_scores) > z_threshold

        # outliers df
        outliers_via_z = self.df[outliers]
        return outliers_via_z

    def filter_outliers(self, outliers_df, key_ID):
        '''
        Filters outliers from df using a the shared key.
        Use in conjuction with outliers_via_z_score or outliers_df_via_IQR methods.
        '''
        mask = ~self.df[key_ID].isin(outliers_df[key_ID]) # if the UDI's in the outliers df match those in the original df, bring back FALSE
        mask
        print(f'length of original df: {len(self.df)}')
        print(f'length of filtered df: {len(self.df[mask])}')
        return self.df[mask]
    
    def outliers_df_via_IQR(self, column):
        # Upper and lower quartiles 
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(.75)
        IQR = Q3 - Q1 

        print(f"Q1 (25th percentile): {Q1}")
        print(f"Q1 (75th percentile): {Q3}")
        print(f"IQR: {IQR}")

        var = self.df[column]

        # Identify outliers 
        outliers = self.df[(var < (Q1 - 1.5 * IQR)) | (var > (Q3 + 1.5 * IQR))]
        return outliers
    
    def drop_column(self, vars):
        '''Returns the filtered df'''
        # df_numeric_vars = failure_data_treating_outliers.drop(['UDI', 'Type'], axis = 1)
        filtered_df = self.df.drop(vars, axis = 1)
        print(f'df columns before filtering: \n {self.df.columns}\n')
        print(f'filtered_df\n: {filtered_df.columns}')
        return filtered_df
    
    def merge(self, df_to_merge, on='UDI', how='left'):
        ''' Merges two DataFrames and gives the length of all DataFrames involved throughout the process'''
        print(f'Length of df: {len(self.df)}')
        print(f'Length of df_to_merge: {len(df_to_merge)}')
        self.df = self.df.merge(df_to_merge, on=on, how=how)
        print(f'Length of new_df: {len(self.df)}')
        return self.df
    
    def rename_colunms(self, col_mapping):
        ''' Renames columns and gives a list of the columns before and after the function is applied'''
        print(f'Before renaming: {list(self.df.columns)}\n')
        self.df = self.df.rename(columns=col_mapping)
        print(f'After renaming: {list(self.df.columns)}\n')  
        return self.df

    def remove_outliers(self, columns, key_ID='UDI', method = ['IQR', 'z_score'], z_threshold = [2,3]):
        '''
        Remove outliers across multiple columns using IQR and return the filtered dataframe.
        '''
        original_len = len(self.df)  # Store the original length of the DataFrame
        filtered_df = self.df  # Start with the original dataframe
        
        for column in columns:
            print(f"Processing column: {column}")
            if method == 'IQR':
                outliers = self.outliers_df_via_IQR(column)
                filtered_df = self.filter_outliers(outliers, key_ID)
            elif method == 'z_score':
                outliers = self.outliers_via_z_score_df(column, z_threshold) # FIX MANISH
                filtered_df = self.filter_outliers(outliers, key_ID)
            # Update the instance's dataframe for each subsequent iteration
            self.df = filtered_df  
            print(f"Finished processing column: {column}\n")

        # Calculate percentage data loss based on the original DataFrame
        len_filtered_df = len(filtered_df)
        percentage_data_loss = ((original_len - len_filtered_df) / original_len) * 100
        print(f'Percentage data loss: {percentage_data_loss:.2f}%')

        return filtered_df, percentage_data_loss
    
    def generate_permutations(self, input_list):
        # Generate all permutations of the given length
        return list(permutations(input_list, len(input_list)))
    
    def remove_outliers_optimised(self, columns, key_ID='UDI', method = ['IQR', 'z_score'], z_threshold = [2,3], suppress_output=False):
        permutations_result = self.generate_permutations(columns)

        combination_percentage_dict = {}

        for i in permutations_result:
            dt = DataTransform(self.df.copy())
            print(f'Testing combination: {list(i)}')

            # If suppress_output is True, suppress print statements in the remove_outliers function
            if suppress_output:
                with contextlib.redirect_stdout(io.StringIO()):
                    filtered_df, percentage_data_loss = dt.remove_outliers(
                        columns=list(i), key_ID=key_ID, method=method, z_threshold=z_threshold
                    )
            else:
                # Normal execution with print statements
                filtered_df, percentage_data_loss = dt.remove_outliers(
                    columns=list(i), key_ID=key_ID, method=method, z_threshold=z_threshold
                )

            # Store the tuples in a dictionary 
            combination_percentage_dict[i] = percentage_data_loss

        # Get the combination with the lowest percentage
        min_combination = min(combination_percentage_dict, key=combination_percentage_dict.get)
        min_value = combination_percentage_dict[min_combination]

        # Create a df to print the results 
        combinations_results_df = pd.DataFrame(list(combination_percentage_dict.items()), columns=['combinations', 'percentage_data_loss'])

        # Reapply the best combination on the original dataframe
        dt = DataTransform(self.df.copy())
        if suppress_output:
            with contextlib.redirect_stdout(io.StringIO()):
                best_filtered_df, _ = dt.remove_outliers(columns=list(min_combination), key_ID=key_ID, method=method, z_threshold=z_threshold)
        else:
            # Normal execution with print statements
            best_filtered_df, _ = dt.remove_outliers(columns=list(min_combination), key_ID=key_ID, method=method, z_threshold=z_threshold)

        # Output the result
        print(f'\nResults:\n {combinations_results_df}')
        print(f'Combination with the lowest data loss: {min_combination}')
        print(f'Lowest percentage data loss: {min_value:.2f}%')

        # Return the best filtered dataframe
        return best_filtered_df, min_combination, combination_percentage_dict
import os

class DataPreprocessing(DataFrameInfo, DataTransform):
    def __init__(self, file_name):
        self.file_name = file_name
        self.file_path = self.find_file_path(file_name)
        self.df = self.load_dataframe(self.file_path)

        # Inherit Methods from Parent Classes
        DataTransform.__init__(self, self.df)
        DataFrameInfo.__init__(self, self.df)

    def find_file_path(self, file_name):
        # Assuming the file is in the same directory as the script or notebook
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, file_name)
        if os.path.exists(file_path):
            print(f"File found: {file_path}")
            return file_path
        else:
            raise FileNotFoundError(f"File '{file_name}' not found in directory '{current_dir}'.")
    
    def load_dataframe(self, file_path):
        # Load the DataFrame from the specified file (assuming it's a CSV for this example)
        try:
            df = pd.read_csv(file_path)
            print("DataFrame loaded successfully.")
            return df
        except Exception as e:
            raise Exception(f"Error loading DataFrame from '{file_path}': {e}")
    
    def initial_load_and_clean_data(self, drop_columns, convert_column_into_dummy_var): # consider renaming columns here 
        '''
        Performs initial cleaning when loading data. 
        Reads csv, drop columns, add dummy variables.
        '''
        print("Executing: Step 1 - Loading and Cleaning Data")
        # self.df = pd.read_csv(file_path)
        print('\nDropping Column')
        self.df = self.drop_column(vars=drop_columns)
        print('\nCreating Dummy Variables')
        # Convert `Type` into dummies & join them to the original dataframe
        dummy_vars = self.create_dummies_from_column(convert_column_into_dummy_var)
        self.df = self.concat_dataframes(dummy_vars)
        print(f'\nColumns of New DataFrame: \n{self.df.columns}')
        print(f'\nFirst row: \n{self.df.head(1)}')
        print("\nCompleted: Step 1 - Data Loaded and Cleaned")
        return self.df

    def impute_missing_values(self, imputation_dict):
        print("\nExecuting: Step 2 - Imputing Missing Values")        
        for column, method in imputation_dict.items():
            self.df[column] = self.impute_column(column_name=column, method=method)
        
        print("Completed: Step 2 - Missing Values Imputed")
        return self.df

    def treat_skewness(self, skew_column, rename_new_column): # redundant 
        print("\nExecuting: Step 3 - Treating for Skewness")
        self.df[rename_new_column] = self.yeojohnson(skew_column)
        print("Completed: Step 3 - Skewness Treated")
        return self.df
    
    def run_diagnostics(self): # not sure if this is needed yet 
        print('Running Diagnostics')
        print(f'Percentage of null values: {self.percentage_of_null()}')
        print(f'\nShape of DataFrame: {self.return_shape()}')
        print(f'\nColumn names: {self.column_names()}')
        # print(f'\nColumn names: {self.df.columns}')
        # print(f'\nLength of new df: {len(self.df)}')


#Step 1 - Inital load and cleaning
preprocessing = DataPreprocessing(file_name='failure_data.csv')    
failure_data = preprocessing.initial_load_and_clean_data(drop_columns=['Unnamed: 0', 'Product ID'], convert_column_into_dummy_var='Type') 
print(f"\nCheck Step 1\nPercentage of Null Values for each column after imputation: \n{preprocessing.percentage_of_null()}")
# vars = ['Air temperature [K]', 'Process temperature [K]','Tool wear [min]' ]
# step_1 = print(failure_data[vars].describe())

print('##############################################################################')
# Step 2 - Impute missing values 
imputation_dict = {
    'Air temperature [K]': 'median',
    'Process temperature [K]': 'mean',
    'Tool wear [min]': 'median'
}
failure_data = preprocessing.impute_missing_values(imputation_dict)
# print(f"\nCheck Step 2\nPercentage of Null Values for each column after imputation: \n{preprocessing.percentage_of_null()}")
# print(step_1)
# print(failure_data[vars].describe())
# print(failure_data.columns)
print('##############################################################################')
# Step 3 - Treat skewness in 'Rotational Speed [rpm]'
print(f"Skew Test before treatement: {preprocessing.skew_test('Rotational speed [rpm]')}")
failure_data = preprocessing.treat_skewness(skew_column='Rotational speed [rpm]', rename_new_column='rotational_speed_normalised') # make the print statements part of the decision
print(f"\nSkew Test after treatement: {preprocessing.skew_test('rotational_speed_normalised')}")
print('##############################################################################')
# Step 4 - Remove outliers
outlier_columns = ['rotational_speed_normalised', 'Torque [Nm]', 'Process temperature [K]'] # wrong outliers 
column_stats_before_removing_outliers = preprocessing.describe_statistics(columns=outlier_columns) # print later for comparison
print('\n')
failure_data, _, _, = preprocessing.remove_outliers_optimised(outlier_columns,key_ID='UDI', method='IQR', suppress_output=True)
print('\nBefore Removing outliers:')
print(column_stats_before_removing_outliers)
print('\nAfter Removing Outliers:')
print(failure_data[outlier_columns].describe())
print('##############################################################################')
failure_data = preprocessing.run_diagnostics() # BUG - this does not work, maybe an issue with the class contruction
print('Expected length of final df should be 9866')
# print(failure_data)





# if __name__ == '__main__':
#     # Step 1 - Main workflow
#     failure_data = load_and_clean_data(file_path="failure_data.csv", drop_columns=['Unnamed: 0', 'Product ID'], convert_column_into_dummy_var='Type') 
#     print('##############################################################################')
#     # Step 2 - Impute missing values
#     imputation_dict = {
#         'Air temperature [K]': 'median',
#         'Process temperature [K]': 'mean',
#         'Tool wear [min]': 'median'
#     }
#     failure_data = impute_missing_values(failure_data, imputation_dict)
#     info = DataFrameInfo(failure_data)
#     print(f"\nCheck Step 2\nPercentage of Null Values for each column after imputation: \n{info.percentage_of_null()}")
#     print('##############################################################################')
#     # Step 3 - Treat skewness in 'Rotational Speed [rpm]'
#     info = DataFrameInfo(failure_data)
#     print(f"Skew Test before treatement: {info.skew_test('Rotational speed [rpm]')}")
#     failure_data = treat_skewness(failure_data, skew_column='Rotational speed [rpm]', rename_new_column='rotational_speed_normalised') # make the print statements part of the decision
#     info = DataFrameInfo(failure_data)
#     print(f"\nSkew Test after treatement: {info.skew_test('rotational_speed_normalised')}")
#     print('##############################################################################')
#     # Step 4 - Remove outliers
#     print('Before Removing outliers:')
#     print(failure_data[['Rotational speed [rpm]', 'Torque [Nm]', 'Process temperature [K]']].describe())
#     # print('\n')
#     outlier_columns = ['Rotational speed [rpm]', 'Torque [Nm]', 'Process temperature [K]']
#     failure_data = remove_outliers(failure_data, outlier_columns, key_id='UDI', method='IQR')
#     print('After Removing Outliers:')
#     print(failure_data[['Rotational speed [rpm]', 'Torque [Nm]', 'Process temperature [K]']].describe())
#     print('##############################################################################')
#     print('DataFrame Diagnostics Post Preproccessing:\n')
#     print(run_diagnostics(failure_data))


