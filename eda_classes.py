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
    
    def describe_statistics(self):
        return self.df.describe().loc[['mean', 'std', '50%']]
    
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

# Plotter Class
class Plotter:
    def __init__(self, df):
        self.df = df
        # self.data_info = data_info
    
    def histplot(self, column, kde=False, ax=None):
        if ax is None:
            ax = plt.gca()  # Get current active axis if none is provided
        sns.histplot(self.df[column], kde=kde, ax=ax)
        ax.set_title(f'Histogram for {column}')

    def histogram_and_skew_sub_plots(self, variable_list, DataFrameInfo_instance, num_cols=3):

        #fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        num_vars = len(variable_list)
        #num_cols = 3  # Define number of columns for the subplot grid
        num_cols = num_cols  # Define number of columns for the subplot grid
        num_rows = math.ceil(num_vars / num_cols)  # Calculate number of rows needed

        # fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 5))  # Adjust the figsize as needed
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(16, num_rows * 5))
        axs = axs.flatten()  # Flatten the 2D array of axes to 1D for easy iteration

        for idx, column in enumerate(variable_list):
            ax = axs[idx]
            # Plot the histogram using the Plotter class's histplot method
            self.histplot(column, kde=True, ax=ax)
            # Calculate skewness using the DataInfo class
            skew_value = DataFrameInfo_instance.skew_test(column_name=column)
            ax.set_title(f'Skew: {skew_value:.2f}')

        # If there are any unused subplots, hide them
        for j in range(idx + 1, len(axs)):
            fig.delaxes(axs[j]) # deletes any unused subplots
        plt.tight_layout()
        plt.show()
        
    
    def plot_qq(self, column, ax=None):
        if ax is None:
            ax = plt.gca()  # Get current active axis if none is provided
        qq_plot= qqplot(self.df[column], scale=1, line ='q', ax=ax)
        plt.title(f'Q-Q plot for {column}')
        plt.show()
    
    def scatter(self, column_name):
        scatter_plot = sns.scatterplot(self.df[column_name])
        plt.show()
        
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

    
    def histograms_with_z_score_bounds(self, vars_list):

        # Define Z-score thresholds
        z_thresholds = [2, 3]

        # Create a figure with subplots in a single row
        num_vars = len(vars_list)
        fig, axs = plt.subplots(1, num_vars, figsize=(num_vars * 6, 6))  # Adjust the figsize as needed

        for idx, var in enumerate(vars_list):
            # Calculate Z-scores for the variable
            z_scores = (self.df[var] - self.df[var].mean()) / self.df[var].std()
            
            # Calculate the percentage of data loss due to each threshold
            loss_z2 = (z_scores.abs() > 2).mean() * 100
            loss_z3 = (z_scores.abs() > 3).mean() * 100
            
            ax = axs[idx]
            
            # Plot the histogram and KDE with a lighter shade of blue for non-outliers
            sns.histplot(data=self.df, x=var, kde=True, color='cornflowerblue', label='Data', stat="density", ax=ax)
            
            # Plot the KDE to get the y-values for the fill_between function
            kde = sns.kdeplot(data=self.df[var], color='blue', ax=ax)
            
            # Calculate the mean and standard deviation
            mean = self.df[var].mean()
            std = self.df[var].std()

            # Draw dotted lines for Z-score thresholds and fill the areas for outliers
            for z in z_thresholds:
                # Calculate positions of Z-score thresholds
                lower_bound = mean - z * std
                upper_bound = mean + z * std
                
                # Plot vertical dotted lines
                ax.axvline(lower_bound, color='black', linestyle='dotted', linewidth=1.5)
                ax.axvline(upper_bound, color='black', linestyle='dotted', linewidth=1.5)
                
                # Get the KDE values for filling
                kde_y = kde.get_lines()[0].get_ydata()
                kde_x = kde.get_lines()[0].get_xdata()
                
                # Fill the areas representing outliers with a darker shade
                ax.fill_between(kde_x, kde_y, where=((kde_x <= lower_bound) | (kde_x >= upper_bound)), 
                                color='darkblue', alpha=0.7, label=f'Outliers (|Z| > {z})')
                
                # Further lower the Z-score labels for better readability
                if var in ['Rotational speed [rpm]', 'Torque [Nm]']:
                    y_position = max(kde_y) * 0.40  # Lower the position more for these variables
                else:
                    y_position = max(kde_y) * 0.85  # Keep higher position for others
                
                # Add bold, centered text labels for Z-scores as absolute values
                ax.text(lower_bound, y_position, f'|Z| = {z}', horizontalalignment='center', 
                        color='black', weight='bold', fontsize=10)
                ax.text(upper_bound, y_position, f'|Z| = {z}', horizontalalignment='center', 
                        color='black', weight='bold', fontsize=10)
            
            # Adding title and labels
            ax.set_title(f'Histogram of {var}')
            ax.set_xlabel(var)
            ax.set_ylabel('Density')
            
            # Move the data loss statistics box down
            ax.text(0.95, 0.65, f'Data Loss:\n|Z| > 2: {loss_z2:.2f}%\n|Z| > 3: {loss_z3:.2f}%', 
                    transform=ax.transAxes, fontsize=10, verticalalignment='top', 
                    horizontalalignment='right', weight='bold', bbox=dict(facecolor='white', alpha=0.8))
            
            # Add the legend
            ax.legend(loc='upper right', fontsize='8', frameon=True, shadow=True, borderpad=1)

        plt.tight_layout()
        plt.show()
    def correlation_heatmap(self, figsize=(12, 10), threshold=0, ax=None):
        """
        Generates a correlation heatmap, optionally filtered by a correlation threshold.
        
        Parameters:
        - figsize: tuple, size of the figure.
        - threshold: float, correlations lower than this value will not be shown.
        - ax: matplotlib.axes, specify the axes for plotting when using subplots.
        """
        corr_matrix = self.df.corr()

        # Apply threshold by setting values below the threshold to NaN (to not show them)
        filtered_corr_matrix = corr_matrix.copy()
        filtered_corr_matrix[abs(filtered_corr_matrix) < threshold] = None

        # Create heatmap on the provided axes (or a new figure if ax is None)
        if ax is None:
            plt.figure(figsize=figsize)
            sns.heatmap(filtered_corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
            plt.title('Correlation Heatmap')
            plt.show()
        else:
            sns.heatmap(filtered_corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
            ax.set_title('Correlation Heatmap')

##########################################################################################
# Step 1 
failure_data = pd.read_csv("failure_data.csv")

dt = DataTransform(failure_data)
failure_data = dt.drop_column(vars=['Unnamed: 0', 'Product ID'])

print(failure_data.head)

# Convert `Type` into dummies & join them onto our original df 
dt = DataTransform(failure_data)
type_as_dummies = dt.create_dummies_from_column('Type')
failure_data = dt.concat_dataframes(type_as_dummies)
failure_data.info()
##########################################################################################
# step 2 - Imputing missing values 
# Impute NULL values of Air temperature 
print('Executing: Step 2 Imputing missing values')
dt = DataTransform(failure_data)
failure_data['Air temperature [K]'] = dt.impute_column(column_name='Air temperature [K]', method='median')

# Impute NULL values of `Process temperature [K]` using the median
dt = DataTransform(failure_data)
failure_data['Process temperature [K]'] = dt.impute_column(column_name='Process temperature [K]', method='mean')

# Impute NULL values of `Tool wear [min]` using the median 
dt = DataTransform(failure_data)
failure_data['Tool wear [min]'] = dt.impute_column(column_name='Tool wear [min]', method='median')

# Prove that the new transformed data contails no missing values 
info_df_without_null = DataFrameInfo(failure_data) # create a print statement here 
info_df_without_null.percentage_of_null()
print('Completed: Step 2 Imputing missing values')
##########################################################################################
# Step 3 - treating for skewness in `rotational speed` # create a print statement here 
# Applying Yeo-Johnson Transformation to `Rotational Speed [rpm]`
print('Executing: Step 3 Treating for Skewness')
dt = DataTransform(failure_data)
failure_data['rotational_speed_normalised'] = dt.yeojohnson('Rotational speed [rpm]')
print('Completed: Step 3 Treating for Skewness')
##########################################################################################
#Step 4: removing outliers 
print('Executing: Step 4 Removing Outliers')
dt = DataTransform(failure_data)
failure_data, min_combination, combination_percentage_dict = dt.remove_outliers_optimised(columns=['Rotational speed [rpm]','Torque [Nm]','Process temperature [K]'], 
                                                                                                  key_ID='UDI', 
                                                                                                  method='IQR',
                                                                                                  suppress_output=True)
print('Completed: Step 4 Removing Outliers')
##########################################################################################
# Step 5: Dropping columns based on Collinearity
# possibly drop `RNF` based on chi-squared tests

##########################################################################################
# proving the changes 

print('\n')
print('\n')
print('\nChecks:')

info = DataFrameInfo(failure_data)
print(f'Percentage of null values: {info.percentage_of_null()}')
print(f'Shape of DataFrame: {info.return_shape()}')
print(f'Column names: {info.column_names()}')

# visual check of the skewness transformation 

plott = Plotter(failure_data)
plott.histogram_and_skew_sub_plots(variable_list=['rotational_speed_normalised'], DataFrameInfo_instance=DataFrameInfo(failure_data)) # improve function 

# Visual check that outliers have been removed 
# plott = Plotter(failure_data)

# List of continuous variables of interest
continous_variables = ['Air temperature [K]', 'Process temperature [K]', 'rotational_speed_normalised', 'Torque [Nm]', 'Tool wear [min]']
plott.boxplots(variable_list=continous_variables)


