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

########################################################################
class DataTransform:
    def __init__(self, df):
        self.df = df

    def filter_outliers(self, outliers_df, key_ID):
        '''
        Filters outliers from df using a shared key.
        Use in conjunction with outliers_via_z_score func.
        '''
        mask = ~self.df[key_ID].isin(outliers_df[key_ID])  # Exclude rows where the key matches
        filtered_df = self.df[mask]
        print(f'Length of original df: {len(self.df)}')
        print(f'Length of filtered df: {len(filtered_df)}')
        return filtered_df

    def outliers_df_via_IQR(self, column):
        '''
        Identify outliers in a column using the IQR method.
        '''
        # Calculate upper and lower quartiles
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1

        print(f"Q1 (25th percentile) for {column}: {Q1}")
        print(f"Q3 (75th percentile) for {column}: {Q3}")
        print(f"IQR for {column}: {IQR}")

        # Identify outliers
        outliers = self.df[(self.df[column] < (Q1 - 1.5 * IQR)) | (self.df[column] > (Q3 + 1.5 * IQR))]
        return outliers
    
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

    # static method
    
    def generate_permutations(self, input_list):
        # Generate all permutations of the given length
        return list(permutations(input_list, len(input_list)))

    def remove_outliers_optimised(self, columns, key_ID='UDI', method = ['IQR', 'z_score'], z_threshold = [2,3]):

        permutations_result = self.generate_permutations(columns)

        combination_percentage_dict = {}

        for i in permutations_result:
            dt= DataTransform(self.df.copy())
            print(f'Testing combination: {list(i)}')
            filtered_df, percentage_data_loss = dt.remove_outliers(columns=list(i), key_ID=key_ID, method=method, z_threshold=z_threshold)

            # store the tuples in a dictionary 
            combination_percentage_dict[i] = percentage_data_loss

        # get the combination with the lowest percentage
        min_combination = min(combination_percentage_dict, key=combination_percentage_dict.get)
        min_value = combination_percentage_dict[min_combination]

        # Output the result
        print(f'Combination with the lowest data loss: {min_combination}')
        print(f'Lowest percentage data loss: {min_value:.2f}%')
        
        # Reapply the best combination on the original dataframe
        dt = DataTransform(self.df.copy())
        best_filtered_df, _ = dt.remove_outliers(columns=list(min_combination), key_ID=key_ID, method=method, z_threshold=z_threshold)

        # Return the best filtered dataframe
        return best_filtered_df, min_combination
    
################################################################################################
failure_data_cleaned_unskewed = pd.read_csv('failure_data_step_3_skew_transformations.csv')
failure_data_cleaned_unskewed.head()
dt = DataTransform(failure_data_cleaned_unskewed)
IQR_filtered_df, min_combination = dt.remove_outliers_optimised(columns=['Process temperature [K]', 'Torque [Nm]', 'Rotational speed [rpm]'], key_ID='UDI', method='IQR')

print('\n Results:')
print(len(IQR_filtered_df))
print(min_combination)


