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
from itertools import permutations
import contextlib
import io
import os

class LoadData:
    def __init__(self, file_name):
        """
        Initialize with the file name and load the data as a pandas DataFrame.
        """
        self.file_name = file_name
        self.df = self.load_dataframe()

    def find_file_path(self):
        """
        Locate the file in the current directory or subdirectory.
        Raise an error if the file is not found.
        """
        current_dir = os.getcwd()  # Get current working directory where the script is running
        file_path = os.path.join(current_dir, self.file_name)  # Form full path

        if os.path.exists(file_path):  # Check if file exists
            print(f"File found: {file_path}")
            return file_path
        else:
            raise FileNotFoundError(f"File '{self.file_name}' not found in directory '{current_dir}'.")

    def load_dataframe(self):
        """
        Load the file as a pandas DataFrame.
        """
        file_path = self.find_file_path()  # Find the file path
        try:
            df = pd.read_csv(file_path)  # Load the CSV into a DataFrame
            print(f"DataFrame loaded successfully from {file_path}.")
            return df
        except Exception as e:
            raise Exception(f"Error loading DataFrame from '{file_path}': {e}")


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
    
    def impute_missing_values(self, imputation_dict):
        print("\nExecuting: Imputing Missing Values")
        
        for column, method in imputation_dict.items():
            # Count the number of NULL values before imputation
            null_count_before = self.df[column].isna().sum()
            print(f'Number of NULL values in {column} before imputation: {null_count_before}')

            # Choose the imputation method
            if method == 'mean':
                impute_value = self.df[column].mean()
            elif method == 'median':
                impute_value = self.df[column].median()
            elif method == 'mode':
                impute_value = self.df[column].mode()[0]
            else:
                raise ValueError(f"Invalid method for {column}. Must be 'mean', 'median', or 'mode'.")

            # Instead of inplace=True, reassign the column explicitly
            self.df[column] = self.df[column].fillna(impute_value)

            # Count the number of NULL values after imputation
            null_count_after = self.df[column].isna().sum()
            print(f'Number of NULL values in {column} after imputation: {null_count_after}')
        
        print("Completed: Imputation of Missing Values")
        return self.df

    def treat_skewness(self, column_name, normalied_column_name, method='log_transform'):
        """
        Applies the specified skewness treatment method to a column.
        Returns the orignal df with the addion of the normalised column.

        Available methods:
        - 'log_transform': Log transformation (for positive values only).
        - 'boxcox': Box-Cox transformation (for positive values only).
        - 'yeojohnson': Yeo-Johnson transformation (for both positive and negative values).
        """
        # Check the method and apply the corresponding transformation
        if method == 'log_transform':
            print(f"Applying log transform to {column_name}")
            # Applying log transformation, making sure only positive values are transformed
            self.df[normalied_column_name] = self.df[column_name].apply(lambda x: np.log(x) if x > 0 else 0)

        elif method == 'boxcox':
            print(f"Applying Box-Cox transform to {column_name}")
            # Box-Cox requires positive values, add a small constant to avoid zero or negative values
            self.df[normalied_column_name], _ = stats.boxcox(self.df[column_name])

        elif method == 'yeojohnson':
            print(f"Applying Yeo-Johnson transform to {column_name}")
            # Yeo-Johnson can handle both positive and negative values
            self.df[normalied_column_name], _ = stats.yeojohnson(self.df[column_name])

        else:
            raise ValueError(f"Unknown method '{method}'. Choose from 'log_transform', 'boxcox', or 'yeojohnson'.")

        print(f"{method} applied to {column_name}.")
        # normalised_var = self.df[normalied_column_name]
        return self.df
    
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
        print(f'Column List: \n {self.df.columns}\n')
        print(f'Column List After dropping df\n: {filtered_df.columns}')
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
    
    def remove_outliers_with_optimiser(self, columns, key_ID='UDI', method = ['IQR', 'z_score'], z_threshold = [2,3], suppress_output=False):
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
    
class Plotter:
    def __init__(self, df):
        self.df = df
        # self.data_info = data_info
    
    def histplot(self, column, kde=False, ax=None):
        if ax is None:
            ax = plt.gca()  # Get current active axis if none is provided
        sns.histplot(self.df[column], kde=kde, ax=ax)
        ax.set_title(f'Histogram for {column}')
        plt.show

    def skew_test(self, column_name):
        return self.df[column_name].skew()

    def histogram_and_skew_sub_plots(self, variable_list, num_cols=3):

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
            skew_value = self.skew_test(column_name=column)
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

class Models: 
    def __init__(self, df):
            self.df = df 

    @staticmethod
    def r_squared(linear_model, model_name):
          r2 = linear_model.rsquared
          print(f"{model_name}: {r2}")
          
    @staticmethod
    def VIF(linear_model, model_name):
         r2 = linear_model.rsquared
         print(f'{model_name}: {1/(1-r2)}')
         return 1/(1-r2)

    def chi_squared_test_df(self, binary_cols):
        # Store results in a list
        results = []

        # Loop through all combinations of columns
        for col1, col2 in itertools.combinations(binary_cols, 2):
            try:
                # Filter rows where both col1 and col2 are not NaN or invalid
                valid_rows = self.df[[col1, col2]].dropna()

                # Create a contingency table for the two columns
                contingency_table = pd.crosstab(valid_rows[col1], valid_rows[col2])
                
                # Perform the Chi-Squared test
                chi2_stat, p_val, dof, expected = chi2_contingency(contingency_table)
                
                # Get the number of valid observations (rows)
                num_obs = len(valid_rows)
                
                # Store the result in the list
                results.append({
                    'Variable 1': col1,
                    'Variable 2': col2,
                    'Chi-Squared Statistic': chi2_stat,
                    'P-Value': p_val,
                    'Degrees of Freedom': dof,
                })
                
            except ValueError as e:
                # Skip the combination if there is an error
                print(f"Skipping combination {col1}, {col2} due to error: {e}")

        # Convert results to a DataFrame for easier viewing
        results_df = pd.DataFrame(results)

        # Display the results
        return results_df
    
    def ols(self, formula='machine_failure ~ air_temperature + process_temperature', model_summary=0):
        """Fit and return the OLS model, optionally printing summary statistics"""
        ols_model = smf.ols(formula, self.df).fit()
        if model_summary == 1:
            print(ols_model.summary2())
        return ols_model

    def logit(self, formula='machine_failure ~ air_temperature + process_temperature', model_summary=0):
        """Fit and return the Logit (logistic regression) model, optionally printing summary statistics"""
        logit_model = smf.logit(formula, self.df).fit()
        if model_summary == 1:
            print(logit_model.summary2())
        return logit_model

    def plot_model_curves(self, predictor_vars, target_var='machine_failure', model='ols', combine_plots=0, ncols=3, standardize=False):
        """Plot OLS or Logit model regression curves for each predictor variable or combine all into one plot."""
        
        # Optionally standardize the variables
        if standardize:
            scaler = StandardScaler()
            self.df[predictor_vars] = scaler.fit_transform(self.df[predictor_vars])

        # Fit the chosen model: OLS or Logit
        formula = f'{target_var} ~ ' + ' + '.join(predictor_vars)
        if model == 'ols':
            model_fit = self.ols(formula)
        elif model == 'logit':
            model_fit = self.logit(formula)
        else:
            raise ValueError("Model must be 'ols' or 'logit'.")

        if combine_plots == 1:
            # Combine all plots into a single plot
            plt.figure(figsize=(10, 6))
            for variable in predictor_vars:
                # Generate a sequence of values over the range of the selected variable
                x_values = np.linspace(self.df[variable].min(), self.df[variable].max(), 100)

                # Create a DataFrame for prediction, setting other variables to their mean dynamically
                predict_data = pd.DataFrame({
                    var: [self.df[var].mean()] * len(x_values) for var in predictor_vars
                })

                # Ensure that only the specific variable changes
                predict_data[variable] = x_values

                # Predict the target variable using the chosen model
                predict_data['predicted_values'] = model_fit.predict(predict_data)

                # Plot the regression/logit curve on the same plot
                plt.plot(x_values, predict_data['predicted_values'], label=f'{variable}')

                # Add data label to the end of each line
                plt.text(x_values[-1], predict_data['predicted_values'].iloc[-1], f'{variable}', 
                         fontsize=9, verticalalignment='center')

            # Customize combined plot
            plt.title(f'{model.upper()} Regression Curves')
            plt.xlabel('Standardized Predictor Variables' if standardize else 'Predictor Variables')
            plt.ylabel(f'Predicted {target_var}')
            plt.grid(True)
            plt.legend(loc='upper left', fontsize=9)  # Adjust legend for clarity
            plt.tight_layout()
            plt.show()

        else:
            # Calculate number of plots and dynamic layout for individual plots
            n_plots = len(predictor_vars)
            nrows = math.ceil(n_plots / ncols)  # Dynamically determine the number of rows

            # Create subplots with dynamic layout
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 6 * nrows))
            axes = axes.flatten()  # Flatten the axes array for easy iteration

            for i, variable in enumerate(predictor_vars):
                # Generate a sequence of values over the range of the selected variable
                x_values = np.linspace(self.df[variable].min(), self.df[variable].max(), 100)

                # Create a DataFrame for prediction, setting other variables to their mean dynamically
                predict_data = pd.DataFrame({
                    var: [self.df[var].mean()] * len(x_values) for var in predictor_vars
                })

                # Ensure that only the specific variable changes
                predict_data[variable] = x_values

                # Predict the target variable using the chosen model
                predict_data['predicted_values'] = model_fit.predict(predict_data)

                # Plot the OLS or Logit regression curve in the subplot
                axes[i].plot(x_values, predict_data['predicted_values'], color='green')
                axes[i].set_title(f'{model.upper()} Regression Curve for {variable}')
                axes[i].set_xlabel('Standardized ' + variable if standardize else variable)
                axes[i].set_ylabel(f'Predicted {target_var}')
                axes[i].grid(True)

                # Add data label to the end of each line
                axes[i].text(x_values[-1], predict_data['predicted_values'].iloc[-1], f'{variable}', 
                             fontsize=9, verticalalignment='center')

            # Remove any extra empty subplots if they exist
            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])

            plt.tight_layout()
            plt.show()