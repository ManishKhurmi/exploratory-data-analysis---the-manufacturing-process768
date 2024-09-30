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


    # def remove_outliers_optimised(self, columns, key_ID='UDI', method = ['IQR', 'z_score'], z_threshold = [2,3]):
    #     permutations_result = self.generate_permutations(columns)

    #     combination_percentage_dict = {}

    #     for i in permutations_result:
    #         dt = DataTransform(self.df.copy())
    #         print(f'Testing combination: {list(i)}')
            
    #         # Suppress print statements in the remove_outliers function
    #         with contextlib.redirect_stdout(io.StringIO()):
    #             filtered_df, percentage_data_loss = dt.remove_outliers(
    #                 columns=list(i), key_ID=key_ID, method=method, z_threshold=z_threshold
    #             )

    #         # Store the tuples in a dictionary 
    #         combination_percentage_dict[i] = percentage_data_loss

    #     # Get the combination with the lowest percentage
    #     min_combination = min(combination_percentage_dict, key=combination_percentage_dict.get)
    #     min_value = combination_percentage_dict[min_combination]

    #     # Create a df to print the results 
    #     combinations_results_df = pd.DataFrame(list(combination_percentage_dict.items()), columns=['combinations', 'percentage_data_loss'])

    #     # Reapply the best combination on the original dataframe
    #     dt = DataTransform(self.df.copy())
    #     with contextlib.redirect_stdout(io.StringIO()):
    #         best_filtered_df, _ = dt.remove_outliers(columns=list(min_combination), key_ID=key_ID, method=method, z_threshold=z_threshold)

    #     # Output the result
    #     print(f'\nResults:\n {combinations_results_df}')
    #     print(f'Combination with the lowest data loss: {min_combination}')
    #     print(f'Lowest percentage data loss: {min_value:.2f}%')

    #     # Return the best filtered dataframe
    #     return best_filtered_df, min_combination, combination_percentage_dict


    # def remove_outliers_optimised(self, columns, key_ID='UDI', method = ['IQR', 'z_score'], z_threshold = [2,3]):

    #     permutations_result = self.generate_permutations(columns)

    #     combination_percentage_dict = {}

    #     for i in permutations_result:
    #         dt= DataTransform(self.df.copy())
    #         print(f'Testing combination: {list(i)}')
    #         filtered_df, percentage_data_loss = dt.remove_outliers(columns=list(i), key_ID=key_ID, method=method, z_threshold=z_threshold)

    #         # store the tuples in a dictionary 
    #         combination_percentage_dict[i] = percentage_data_loss

    #     # get the combination with the lowest percentage
    #     min_combination = min(combination_percentage_dict, key=combination_percentage_dict.get)
    #     min_value = combination_percentage_dict[min_combination]

    #     # create a df to print the results 
    #     combinations_results_df = pd.DataFrame(list(combination_percentage_dict.items()), columns=['combinations', 'percentage_data_loss'])

    #     # Reapply the best combination on the original dataframe
    #     dt = DataTransform(self.df.copy())
    #     best_filtered_df, _ = dt.remove_outliers(columns=list(min_combination), key_ID=key_ID, method=method, z_threshold=z_threshold)

    #     # Output the result
    #     print(f'\nResults:\n {combinations_results_df}')
    #     print(f'Combination with the lowest data loss: {min_combination}')
    #     print(f'Lowest percentage data loss: {min_value:.2f}%')

    #     # Return the best filtered dataframe
    #     return best_filtered_df, min_combination, combination_percentage_dict


    
########################################################################################################################
# Plotter Class
class Plotter:
    def __init__(self, df):
        self.df = df
        # self.data_info = data_info   
    def boxplot(self, column):
        box_plot = sns.boxplot(self.df[column])
        plt.show()
    
    def boxplots(self, variable_list):

        # Number of variables
        num_vars = len(variable_list)

        # Create subplots (1 row, num_vars columns)
        fig, axs = plt.subplots(1, num_vars, figsize=(num_vars * 4, 5))  # Adjust the width and height of the figure

        # Plot each variable
        for idx, i in enumerate(variable_list):
            sns.boxplot(data=self.df, y=i, ax=axs[idx])  # y=i for vertical boxplots
            axs[idx].set_title(f'{i}')
            axs[idx].set_xlabel('')  # Remove x-axis label to save space
            axs[idx].set_ylabel('')  # Remove y-axis label to save space

        plt.tight_layout()
        plt.show()




########################################################################################################################################
########################################################################################################################################
# testing - Hardcoded, z_score_threshold = 2 
# # TODO: Must keep for Appendix 

# # rotational speed 
# failure_data_cleaned_unskewed = pd.read_csv('failure_data_step_3_skew_transformations.csv')

# dt = DataTransform(failure_data_cleaned_unskewed)
# rotational_speed_z_score_outliers = dt.outliers_via_z_score_df('Rotational speed [rpm]', z_threshold=2)
# filtered_rotational_speed_df = dt.filter_outliers(rotational_speed_z_score_outliers, key_ID='UDI')
# len(filtered_rotational_speed_df)

# plott_org = Plotter(failure_data_cleaned_unskewed)

# print('\nRotational speed [rpm] Actual')
# plott_org.boxplot('Rotational speed [rpm]')

# print('Outliers Removed Rotational speed [rpm]')
# plott_filtered_df = Plotter(filtered_rotational_speed_df)
# plott_filtered_df.boxplot('Rotational speed [rpm]')
# print('\n')
# print('\n')

# ################################################################################
# # Torque
# # filtered_rotational_speed_df.head()
# # Filter torque using the new filtered df 

# dt = DataTransform(filtered_rotational_speed_df)
# torque_z_score_outliers = dt.outliers_via_z_score_df('Torque [Nm]', z_threshold = 2)
# filtered_rotational_speed_torque_df = dt.filter_outliers(torque_z_score_outliers, key_ID='UDI')

# print('Torqe actual')
# plott_org.boxplot('Torque [Nm]')

# plott = Plotter(filtered_rotational_speed_torque_df)
# print(len(filtered_rotational_speed_torque_df))
# print('Torque after removing outliers')
# plott.boxplot('Torque [Nm]')

# #####
# # Process Temperature 

# dt = DataTransform(filtered_rotational_speed_torque_df)
# process_temp_z_score_outliers = dt.outliers_via_z_score_df('Process temperature [K]', z_threshold=2)
# filtered_rotational_speed_torque_process_temp_df = dt.filter_outliers(process_temp_z_score_outliers, key_ID='UDI')

# print('Process temperature Acutal')
# plott_org.boxplot('Process temperature [K]')

# print('Process temperature: Outliers Removed')
# plott = Plotter(filtered_rotational_speed_torque_process_temp_df)
# plott.boxplot('Process temperature [K]')
# len(filtered_rotational_speed_torque_process_temp_df)

# len_org_df = len(failure_data_cleaned_unskewed)
# len_filtered_df = len(filtered_rotational_speed_torque_process_temp_df)
# percentage_data_loss = ((len_org_df - len_filtered_df) / len_org_df) * 100
# print(f'Percentage data loss {percentage_data_loss}')

# success, percentage data loss is 11.1% as expected 
##################################################################################################################################
########################################################################################################################################
# testing refactored 

# failure_data_cleaned_unskewed = pd.read_csv('failure_data_step_3_skew_transformations.csv')
# dt = DataTransform(failure_data_cleaned_unskewed)
# IQR_filtered_df, min_combination = dt.remove_outliers_optimised(columns=['Process temperature [K]', 'Torque [Nm]', 'Rotational speed [rpm]'], key_ID='UDI', method='IQR')

# print('\n Results:')
# print(len(IQR_filtered_df))
# print(min_combination)

## z_score
# z_filtered_df, percentage_data_loss = dt.remove_outliers(columns=['Rotational speed [rpm]','Torque [Nm]','Process temperature [K]'], key_ID='UDI', method='z_score', z_threshold=2)
# plott = Plotter(z_filtered_df)
# plott.boxplots(['Rotational speed [rpm]','Torque [Nm]','Process temperature [K]'])

## IQR
# IQR_filtered_df, percentage_data_loss = dt.remove_outliers(columns=['Rotational speed [rpm]','Torque [Nm]','Process temperature [K]'], key_ID='UDI', method='IQR')
# plott = Plotter(IQR_filtered_df)
# plott.boxplots(['Rotational speed [rpm]','Torque [Nm]','Process temperature [K]'])

# IQR - data loss 4.75%
## success 
########################################################################################################################################
########################################################################################################################################
# # Optimising 

failure_data_cleaned_unskewed = pd.read_csv('failure_data_step_3_skew_transformations.csv')
dt = DataTransform(failure_data_cleaned_unskewed)

IQR_best_filtered_df, min_combination, combination_percentage_dict = dt.remove_outliers_optimised(columns=['Rotational speed [rpm]','Torque [Nm]','Process temperature [K]'], 
                                                                                                  key_ID='UDI', 
                                                                                                  method='IQR',
                                                                                                  suppress_output=True)



# print(f'\nResults:\n {combination_percentage_dict}')
# print(f'Best combination: {min_combination}')
# print(f'df with minimal data loss: {len(IQR_best_filtered_df)}')



# great success 

# display(combination_percentage_dict)

# Design 

# results table / dict 
# best combination 
# length of optimal df 

# combinations_results_df = pd.DataFrame(list(combination_percentage_dict.items()), columns=['combinations', 'percentage_data_loss'])
# print(combinations_results_df)