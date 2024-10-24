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
    def __init__(self, file_name: str) -> None: 
        """
        Initialize with the file name and load the data as a pandas DataFrame. 

        Args:
        file_name (str): The name of the file to load data from.

        Example:
        file_name = 'failure_data.csv'
        """
        self.file_name = file_name
        self.df = self.load_dataframe()

    def find_file_path(self):
        """
        Locate the file in the 'data' directory.
        Raise an error if the file is not found.
        """
        # Directly look in the 'data' directory relative to the project's root
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        file_path = os.path.join(data_dir, self.file_name)

        if os.path.exists(file_path):  # Check if file exists
            print(f"File found: {file_path}")
            return file_path
        else:
            raise FileNotFoundError(f"File '{self.file_name}' not found in the 'data' directory.")

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
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df 
    
    def return_shape(self) -> str:
        return str(self.df.shape) 
    
    def return_info(self) -> None:
        return self.df.info()
    
    def return_first_row(self) -> pd.Series:
         return self.df.iloc[0]

    def data_type(self) -> None:
        print(self.df.dtypes)
    
    def describe_statistics(self, columns: list[str]) -> pd.DataFrame:
        return self.df[columns].describe()
    
    def unique_value_count(self, column_names: list[str]) -> pd.Series:
        return self.df[column_names].nunique()
    
    def percentage_of_null(self) -> pd.Series:
        percentage_of_null = self.df.isnull().sum() / len(self.df) * 100  
        return percentage_of_null
    
    def are_all_observations_unique(self, column_name: list[str]) -> pd.Series:
        print(f'The {column_name} column contains only unique rows: {len(self.df) == self.df[column_name].nunique()}')
    
    def normal_test(self, column_name: str) -> None:
        stat, p = normaltest(self.df[column_name], nan_policy = 'omit')
        print('Statistics=%.3f, p=%.3f' % (stat, p))

    def skew_test(self, column_name: str) -> float:
        return self.df[column_name].skew()

    def print_mean(self, column_name) -> None:
        print(f'The mean of {column_name} is {self.df[column_name].mean()}')
    
    def print_median(self, column_name) -> None:
        print(f'The median of {column_name} is {self.df[column_name].median()}')

    def column_names(self) -> pd.Index:
        return self.df.columns

    def continous_variables(self) -> list[str]:
        '''    
        Identifies and returns a list of continuous variables in the DataFrame.
        A variable is considered continuous if it has more than two unique values.
        Returns:
            list: A list of column names representing continuous variables.
        '''
        continous_variables = []
        for i in self.df.columns:
            if self.df[i].nunique() > 2:
                continous_variables.append(i)
        return continous_variables
    
    def z_score_info(self, z_scores: pd.Series) -> None:
        ''' 
        Displays the number and percentage of outliers for z-score thresholds of ±2 and ±3.
        Useful for understanding data loss when filtering outliers based on these thresholds.
        
        Example usage:
        air_temperature_z_scores = dt.z_score('air_temperature')
        info = DataFrameInfo(failure_data_df)
        info.z_score_info(air_temperature_z_scores)
        '''
        # Z-score Threshold 
        threshold_2 = 2 
        threshold_3 = 3

        outliers_2 = (np.abs(z_scores) > threshold_2).sum() 
        outliers_3 = (np.abs(z_scores) > threshold_3).sum()

        percentage_outliers_thereshold_2 = round(outliers_2/len(z_scores) * 100, 2)
        percentage_outliers_thereshold_3 = round(outliers_3/len(z_scores) * 100, 2)

        print(f"Number of observations with outliers based on z-score threshold ±2: {outliers_2}")
        print(f"Percentage of observations with outliers based on z-score threshold ±2: {percentage_outliers_thereshold_2}")
        print("\n")
        print(f"Number of observations with outliers based on z-score threshold ±3: {outliers_3}")
        print(f"Percentage of observations with outliers based on z-score threshold ±3: {percentage_outliers_thereshold_3}")
    
    def range_df(self, columns: list[str]) -> pd.DataFrame:
        '''Calculate and return the range (minimum and maximum values) for each specified column in the DataFrame.'''
        min_values = self.df[columns].min()
        max_values = self.df[columns].max()

        range_df = pd.DataFrame({
        'Variables': columns,
        'Min': min_values.values,
        'Max': max_values.values
        })

        range_df.set_index('Variables', inplace=True)
        return range_df


class DataTransform:
    def __init__(self, df) -> None:
        self.df = df

    def unique_observations(self, column_name: str) -> np.ndarray:
        '''Retrieve unique values from a specified column in the DataFrame.'''
        return self.df[column_name].unique()
    
    def convert_column_to_category(self, column_name: str) -> pd.DataFrame:
        ''' converts the dtype of column to `category` '''
        self.df[column_name] = pd.Categorical(self.df[column_name])
        return self.df
    
    def create_dummies_from_column(self, column_name: str) -> pd.DataFrame:
        '''Generate dummy/indicator variables from a specified column.'''
        dummies_df = pd.get_dummies(self.df[column_name], dtype=int)
        return dummies_df 

    def concat_dataframes(self, new_df: pd.DataFrame, left_index: bool = True, right_index: bool = True) -> pd.DataFrame:
        '''Concatenate the current DataFrame with another DataFrame'''
        joined_df = pd.concat([self.df, new_df], axis = 1)
        return joined_df
    
    def impute_missing_values(self, imputation_dict: dict[str,str]) -> pd.DataFrame:
        '''Impute missing values in columns based on specified methods.'''
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

    def treat_skewness(self, column_name: str, normalied_column_name: str, method: str ='log_transform') -> pd.DataFrame:
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
    
    def z_score(self, column: str) -> pd.Series: 
        '''Calculates the Z-Scores of a specified column'''
        x = self.df[column] 
        mean= np.mean(x)
        standard_deviation = np.std(x)
        z_scores = (x - mean) / standard_deviation
        return z_scores
    
    def outliers_via_z_score_df(self, column: str, z_threshold: list[float] = [[2,3]]) -> pd.DataFrame:
        '''Returns a df containing the outliers based on Z-score thresholds.'''
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

    def filter_outliers(self, outliers_df: pd.DataFrame, key_ID: str) -> pd.DataFrame:
        '''
        Filters outliers from df using a the shared key.
        Use in conjuction with outliers_via_z_score or outliers_df_via_IQR methods.
        '''
        mask = ~self.df[key_ID].isin(outliers_df[key_ID]) # if the UDI's in the outliers df match those in the original df, bring back FALSE
        mask
        print(f'length of original df: {len(self.df)}')
        print(f'length of filtered df: {len(self.df[mask])}')
        return self.df[mask]
    
    def outliers_df_via_IQR(self, column: str) -> pd.DataFrame:
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
    
    def drop_column(self, vars: list[str]) -> pd.DataFrame:
        '''Returns the filtered df'''
        # df_numeric_vars = failure_data_treating_outliers.drop(['UDI', 'Type'], axis = 1)
        filtered_df = self.df.drop(vars, axis = 1)
        print(f'Column List: \n {self.df.columns}\n')
        print(f'Column List After dropping df\n: {filtered_df.columns}')
        return filtered_df
    
    def merge(self, df_to_merge, on: str ='UDI', how: str ='left'):
        ''' Merges two DataFrames and gives the length of all DataFrames involved throughout the process'''
        print(f'Length of df: {len(self.df)}')
        print(f'Length of df_to_merge: {len(df_to_merge)}')
        self.df = self.df.merge(df_to_merge, on=on, how=how)
        print(f'Length of new_df: {len(self.df)}')
        return self.df
    
    def rename_colunms(self, col_mapping: dict[str, str]) -> pd.DataFrame:
        ''' Renames columns and gives a list of the columns before and after the function is applied'''
        print(f'Before renaming: {list(self.df.columns)}\n')
        self.df = self.df.rename(columns=col_mapping)
        print(f'After renaming: {list(self.df.columns)}\n')  
        return self.df

    def remove_outliers(self, columns: list[str], key_ID: str ='UDI', method: str = ['IQR', 'z_score'], z_threshold: list[float] = [2,3]):
        '''Remove outliers across multiple columns using IQR and return the filtered dataframe.'''
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
    
    def generate_permutations(self, input_list: list[str]) -> list[tuple[str]]:
        '''Generate all permutations of a given list.'''
        return list(permutations(input_list, len(input_list)))
    
    def remove_outliers_with_optimiser(self, columns: list[str], key_ID: str='UDI', method: str = ['IQR', 'z_score'], z_threshold: list[float] = [2,3], suppress_output: bool=False) -> tuple[pd.DataFrame, tuple[str], dict[tuple[str], float]]:
        '''Optimally remove outliers by testing permutations of the specified columns.'''
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
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
    
    def histplot(self, column: str, kde: bool = False, ax = None) -> None:
        '''Plot a histogram for a given column with an optional KDE overlay.'''
        if ax is None:
            ax = plt.gca()  # Get current active axis if none is provided
        sns.histplot(self.df[column], kde=kde, ax=ax)
        ax.set_title(f'Histogram for {column}')
        plt.show

    def skew_test(self, column_name: str) -> float:
        '''Calculate the skewness of a specified column.'''
        return self.df[column_name].skew()

    def histogram_and_skew_sub_plots(self, variable_list: list[str], num_cols: int = 3) -> None:
        '''Create subplots for histograms and skewness of multiple variables.'''
        num_vars = len(variable_list)
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

    def plot_qq(self, column: str, ax=None) -> None:
        '''Generate a Q-Q (Quantile-Quantile) plot for a specified column.'''
        if ax is None:
            ax = plt.gca()  # Get current active axis if none is provided
        qq_plot= qqplot(self.df[column], scale=1, line ='q', ax=ax)
        plt.title(f'Q-Q plot for {column}')
        plt.show()
    
    def scatter(self, column_name: str) -> None:
        ''' Create a scatter plot for a given column.'''
        scatter_plot = sns.scatterplot(self.df[column_name])
        plt.show()
        
    def boxplot(self, column: str) -> None:
        '''Create a boxplot for a specified column.'''
        box_plot = sns.boxplot(self.df[column])
        plt.show()
    
    def boxplots(self, variable_list: list[str]) -> None:
        '''Create boxplots for multiple variables side by side.'''

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
    
    def histograms_with_z_score_bounds(self, vars_list: list[str]) -> None:
        '''Plot histograms with Z-score thresholds for multiple variables.'''

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

    def correlation_heatmap(self, figsize: tuple[int, int] =(12, 10), threshold: float = 0, ax=None) -> None:
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
    def __init__(self, df) -> None:
            self.df = df 

    @staticmethod
    def r_squared(linear_model, model_name: str) -> None:
        '''
        Print the R-squared value of a linear model.
        Args:
            linear_model: Fitted linear model object.
            model_name (str): Name of the model for display purposes.
        '''
        r2 = linear_model.rsquared
        print(f"{model_name}: {r2}")
          
    @staticmethod
    def VIF(linear_model, model_name) -> float:
         '''
         Calculate and print the Variance Inflation Factor (VIF) for a linear model.
         Args:
            linear_model: Fitted linear model object.
            model_name (str): Name of the model for display purposes.
         Returns:
            float: Calculated VIF value.
         '''
         r2 = linear_model.rsquared
         print(f'{model_name}: {1/(1-r2)}')
         return 1/(1-r2)

    def chi_squared_test_df(self, binary_cols: list[str]) -> pd.DataFrame:
        '''
        Perform a Chi-Squared test between pairs of binary columns.
        
        Args:
            binary_cols (List[str]): List of binary column names.
            
        Returns:
            pd.DataFrame: DataFrame containing Chi-Squared test results for each pair.
        '''
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
    
    def ols(self, formula: str ='machine_failure ~ air_temperature + process_temperature', model_summary: int = 0):
        """Fit and return the OLS model, optionally printing summary statistics"""
        ols_model = smf.ols(formula, self.df).fit()
        if model_summary == 1:
            print(ols_model.summary2())
        return ols_model

    def logit(self, formula: str ='machine_failure ~ air_temperature + process_temperature', model_summary: int = 0):
        """Fit and return the Logit (logistic regression) model, optionally printing summary statistics"""
        logit_model = smf.logit(formula, self.df).fit()
        if model_summary == 1:
            print(logit_model.summary2())
        return logit_model
    
    def first_derivative(self, x: float, coefficients: list[float]) -> float :
        """Calculate the first derivative of the logistic regression curve."""
        # This assumes the logistic function f(x) = 1 / (1 + exp(- (b0 + b1 * x)))
        return np.exp(-x) / (1 + np.exp(-x))**2

    def second_derivative(self, x: float, coefficients: list[float]) -> float:
        """Calculate the second derivative of the logistic regression curve."""
        # Second derivative for logistic regression
        return -np.exp(-x) * (1 - np.exp(-x)) / (1 + np.exp(-x))**3

    def find_local_minima(self, x_values: np.ndarray, y_values: np.ndarray) -> tuple[float, float]:
        """Helper function to find local minima in a given set of x and y values."""
        local_min_x = None
        local_min_y = None
        for i in range(1, len(y_values) - 1):
            if y_values[i - 1] > y_values[i] < y_values[i + 1]:
                local_min_x = x_values[i]
                local_min_y = y_values[i]
                break  # Return the first local minimum found
        return local_min_x, local_min_y

    def plot_model_curves(self, predictor_vars: list[str], target_var: str = 'machine_failure', model: str = 'ols', combine_plots: int = 0, 
                        ncols: int = 3, standardize: bool = False, plot_derivatives: bool = False, local_minima: bool = False, 
                        theoretical_strategy: dict[str, list[float]] = None, business_strategy: dict[str, list[float]] = None) -> dict[str, dict[str, tuple[float, float]]]:
        """
        Plot OLS or Logit model regression curves for each predictor variable or combine all into one plot.
        Optionally standardizes the variables and returns the minima coordinates for first and second derivatives.

        Args:
            predictor_vars (List[str]): List of predictor variable names.
            target_var (str, optional): The target variable for regression. Default is 'machine_failure'.
            model (str, optional): The regression model to use ('ols' or 'logit'). Default is 'ols'.
            combine_plots (int, optional): Whether to combine plots for all predictors (1) or not (0). Default is 0.
            ncols (int, optional): Number of columns for subplot grid. Default is 3.
            standardize (bool, optional): Whether to standardize predictor variables. Default is False.
            plot_derivatives (bool, optional): Whether to plot first and second derivatives. Default is False.
            local_minima (bool, optional): Whether to find and mark local minima of derivatives. Default is False.
            theoretical_strategy (Optional[Dict[str, List[float]]], optional): Theoretical threshold values. Default is None.
            business_strategy (Optional[Dict[str, List[float]]], optional): Business-defined threshold values. Default is None.

        Returns:
            Dict[str, Dict[str, Optional[Tuple[float, float]]]]: Coordinates of local minima for derivatives.
        """
        
        # Dictionary to store the scaler for each predictor variable if standardization is applied
        self.scalers = {}

        # Optionally standardize the variables
        if standardize:
            for var in predictor_vars:
                scaler = StandardScaler()
                self.df[[var]] = scaler.fit_transform(self.df[[var]])
                self.scalers[var] = scaler  # Store scaler for descaling later

        # Fit the chosen model: OLS or Logit
        formula = f'{target_var} ~ ' + ' + '.join(predictor_vars)
        if model == 'ols':
            model_fit = self.ols(formula)
        elif model == 'logit':
            model_fit = self.logit(formula)
        else:
            raise ValueError("Model must be 'ols' or 'logit'.")

        # Get the coefficients for the model (used for derivatives)
        coefficients = model_fit.params

        # Initialize minima coordinates dictionary
        minima_coordinates = {var: {'first_derivative_min': None, 'second_derivative_min': None} for var in predictor_vars}

        # If combining all plots into a single plot
        if combine_plots == 1:
            plt.figure(figsize=(10, 6))
            for variable in predictor_vars:
                x_values = np.linspace(self.df[variable].min(), self.df[variable].max(), 100)

                # Create a DataFrame for prediction, setting other variables to their mean
                predict_data = pd.DataFrame({
                    var: [self.df[var].mean()] * len(x_values) for var in predictor_vars
                })
                predict_data[variable] = x_values
                predict_data['predicted_values'] = model_fit.predict(predict_data)

                plt.plot(x_values, predict_data['predicted_values'], label=f'{variable}', color='blue')

                # Plot theoretical and business strategies if provided
                if theoretical_strategy and variable in theoretical_strategy:
                    for x_val in theoretical_strategy[variable]:
                        plt.axvline(x=x_val, color='blue', linestyle='--', label=f'Theoretical Strategy')
                if business_strategy and variable in business_strategy:
                    for x_val in business_strategy[variable]:
                        plt.axvline(x=x_val, color='orange', linestyle='--', label=f'Business Strategy')

                # Compute derivatives and minima if requested
                if plot_derivatives or local_minima:
                    y_first_derivative = self.first_derivative(x_values, coefficients)
                    y_second_derivative = self.second_derivative(x_values, coefficients)

                    if plot_derivatives:
                        plt.plot(x_values, y_first_derivative, label=f'First Derivative', linestyle='--', color='green')
                        plt.plot(x_values, y_second_derivative, label=f'Second Derivative', linestyle=':', color='red')

                    if local_minima:
                        first_min_x, first_min_y = self.find_local_minima(x_values, y_first_derivative)
                        second_min_x, second_min_y = self.find_local_minima(x_values, y_second_derivative)

                        minima_coordinates[variable]['first_derivative_min'] = (first_min_x, first_min_y)
                        minima_coordinates[variable]['second_derivative_min'] = (second_min_x, second_min_y)

                        # Plot local minima
                        plt.scatter(first_min_x, first_min_y, color='orange', label=f'First Derivative Min')
                        plt.scatter(second_min_x, second_min_y, color='purple', label=f'Second Derivative Min')

                plt.text(x_values[-1], predict_data['predicted_values'].iloc[-1], f'{variable}', fontsize=9, verticalalignment='center')

            plt.title(f'{model.upper()} Regression Curves')
            plt.xlabel('Standardized Predictor Variables' if standardize else 'Predictor Variables')
            plt.ylabel(f'Predicted {target_var}')
            plt.grid(True)
            plt.legend(loc='upper left', fontsize=9)
            plt.tight_layout()
            plt.show()

        # Handle separate plots for each variable
        else:
            n_plots = len(predictor_vars)
            nrows = math.ceil(n_plots / ncols)
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 6 * nrows))
            axes = axes.flatten()

            for i, variable in enumerate(predictor_vars):
                x_values = np.linspace(self.df[variable].min(), self.df[variable].max(), 100)
                predict_data = pd.DataFrame({
                    var: [self.df[var].mean()] * len(x_values) for var in predictor_vars
                })
                predict_data[variable] = x_values
                predict_data['predicted_values'] = model_fit.predict(predict_data)

                axes[i].plot(x_values, predict_data['predicted_values'], color='blue', label=f'{variable}')

                # Plot theoretical and business strategies if provided
                if theoretical_strategy and variable in theoretical_strategy:
                    for x_val in theoretical_strategy[variable]:
                        axes[i].axvline(x=x_val, color='blue', linestyle='--', label=f'Theoretical Strategy')
                if business_strategy and variable in business_strategy:
                    for x_val in business_strategy[variable]:
                        axes[i].axvline(x=x_val, color='orange', linestyle='--', label=f'Business Strategy')

                # Compute derivatives and minima if requested
                if plot_derivatives or local_minima:
                    y_first_derivative = self.first_derivative(x_values, coefficients)
                    y_second_derivative = self.second_derivative(x_values, coefficients)

                    if plot_derivatives:
                        axes[i].plot(x_values, y_first_derivative, label=f'First Derivative', linestyle='--', color='green')
                        axes[i].plot(x_values, y_second_derivative, label=f'Second Derivative', linestyle=':', color='red')

                    if local_minima:
                        first_min_x, first_min_y = self.find_local_minima(x_values, y_first_derivative)
                        second_min_x, second_min_y = self.find_local_minima(x_values, y_second_derivative)

                        minima_coordinates[variable]['first_derivative_min'] = (first_min_x, first_min_y)
                        minima_coordinates[variable]['second_derivative_min'] = (second_min_x, second_min_y)

                        # Plot local minima
                        axes[i].scatter(first_min_x, first_min_y, color='orange', label=f'First Derivative Min')
                        axes[i].scatter(second_min_x, second_min_y, color='purple', label=f'Second Derivative Min')

                axes[i].set_title(f'{model.upper()} Regression Curve for {variable}')
                axes[i].set_xlabel('Standardized ' + variable if standardize else variable)
                axes[i].set_ylabel(f'Predicted {target_var}')
                axes[i].grid(True)
                axes[i].text(x_values[-1], predict_data['predicted_values'].iloc[-1], f'{variable}', fontsize=9, verticalalignment='center')

            # Hide any extra axes
            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])

            handles, labels = axes[i].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper right', fontsize=9)
            plt.tight_layout()
            plt.show()

        return minima_coordinates

    def inverse_scale_minima(self, minima_coordinates: dict[str, dict[str, tuple[float, float]]]) -> dict[str, dict[str, tuple[float, float]]]:
        """
        Inverse the scaling applied to the minima coordinates using the data from self.df and the scalers saved in plot_model_curves.

        Args:
        minima_coordinates (dict): Dictionary containing the local minima coordinates for each predictor variable.

        Returns:
        dict: Dictionary with the minima coordinates inversely scaled.
        """
        # Extract predictor variables from the keys of minima_coordinates
        predictor_vars = list(minima_coordinates.keys())

        # Loop through each predictor variable and inverse scale its minima using the stored scalers
        for var in predictor_vars:
            if var not in self.scalers:
                raise ValueError(f"No scaler found for variable '{var}'.")

            scaler = self.scalers[var]  # Retrieve the saved scaler for this variable

            # Inverse transform the minima only if they are not None
            first_min = minima_coordinates[var]['first_derivative_min'][0]
            second_min = minima_coordinates[var]['second_derivative_min'][0]

            # Inverse scale if the minima are not None
            if first_min is not None:
                first_min_unscaled = scaler.inverse_transform([[first_min]])[0][0]
                minima_coordinates[var]['first_derivative_min'] = (first_min_unscaled, minima_coordinates[var]['first_derivative_min'][1])
            if second_min is not None:
                second_min_unscaled = scaler.inverse_transform([[second_min]])[0][0]
                minima_coordinates[var]['second_derivative_min'] = (second_min_unscaled, minima_coordinates[var]['second_derivative_min'][1])

        return minima_coordinates

    def extract_x_value_of_second_derivative(self, minima_coordinates: dict[str, dict[str, tuple[float, float]]]) -> dict[str, list[int]]:
        """
        Returns a dictionary of variables and x-values by extracting the x-value of the second derivative minima,
        rounded to the nearest integer.
        
        Args:
        minima_coordinates (dict): Dictionary containing the local minima coordinates for each predictor variable.

        Returns:
        dict: Dictionary where keys are variable names and values are the rounded x-values of the second derivative.
        """
        strategy_dict = {}

        # Loop through each variable in the minima_coordinates dictionary
        for var, coords in minima_coordinates.items():
            second_derivative_min_x = coords['second_derivative_min'][0]  # Get the x-value of the second derivative

            # Round the x-value to the nearest integer
            if second_derivative_min_x is not None:
                strategy_dict[var] = [round(second_derivative_min_x)]
        
        return strategy_dict

    # Helper function to calculate failure rate after applying a list of filters in a specific order
    def apply_thresholds_in_order(self, strategy_dict: dict[str, list[float]], filter_order: list[str]) -> tuple[pd.DataFrame, float]:
        '''
        Apply a sequence of thresholds to the DataFrame in a specified order.
    
        Args:
            strategy_dict (Dict[str, List[float]]): Dictionary of thresholds for each variable.
            filter_order (List[str]): List specifying the order of variables to apply thresholds.
        
        Returns:
            Tuple[Optional[pd.DataFrame], Optional[float]]: Filtered DataFrame and resulting failure rate.

        '''

        filtered_df = self.df.copy()

        for variable in filter_order:
            threshold = strategy_dict[variable][0]  # Assuming the first element is the threshold
            filtered_df = filtered_df[filtered_df[variable] < threshold]

        filtered_failure_count = filtered_df['machine_failure'].sum()
        filtered_data_points = len(filtered_df)
        if filtered_data_points == 0:
            return None, None  # Avoid division by zero

        filtered_failure_rate = (filtered_failure_count / filtered_data_points) * 100
        return filtered_df, filtered_failure_rate

    def impact_of_strategy(self, strategy_dict: dict[str, list[float]]) -> dict[str, float]:
        ''' 
        Finds the best order of filters to minimize the failure rate. 
        Returns the best order, the failure rate after applying the filters, 
        and other metrics like production loss.
        '''
        # Extract original metrics
        total_data_points = len(self.df)
        original_failure_count = self.df['machine_failure'].sum()
        original_failure_rate = (original_failure_count / total_data_points) * 100
        print(f'Original Failure Rate: {original_failure_rate}')
        print(f'Total Data Points: {total_data_points}')
        
        # Generate all possible orders of filtering
        variables_with_thresholds = [var for var, threshold in strategy_dict.items() if threshold]
        possible_orders = list(permutations(variables_with_thresholds))
        
        # Track the best order and the lowest failure rate
        best_order = None
        lowest_failure_rate = float('inf')
        best_filtered_df = None

        # Evaluate each order of filtering
        for order in possible_orders:
            filtered_df, failure_rate = self.apply_thresholds_in_order(strategy_dict, order)
            if filtered_df is not None and failure_rate < lowest_failure_rate:
                lowest_failure_rate = failure_rate
                best_order = order
                best_filtered_df = filtered_df

        # Calculate production loss
        filtered_data_points = len(best_filtered_df) if best_filtered_df is not None else total_data_points
        production_loss = total_data_points - filtered_data_points
        relative_improvement_percentage = (original_failure_rate - lowest_failure_rate) / original_failure_rate * 100 if original_failure_rate != 0 else 0

        print(f'Best Order of Applying Filters: {best_order}')
        print(f'Original Failure Rate')
        print(f'Lowest Failure Rate After Applying Best Order: {lowest_failure_rate:.2f}%')
        print(f'Production Loss: {production_loss}')
        print(f'Improvement Percentage: {relative_improvement_percentage:.2f}%')

        return {
            'best_order': best_order,
            'original_failure_rate': f'{original_failure_rate:.2f}%',
            'lowest_failure_rate': f'{lowest_failure_rate:.2f}%',  # As a percentage
            'production_loss': production_loss,
            'relative_improvement_percentage': f'{relative_improvement_percentage:.2f}%'  # As a percentage
        }
    @staticmethod
    def present_results(result_dict: dict[str, str]) -> str:
        """
        Presents the model results in a more readable format.
        
        Parameters:
        result_dict (dict): Dictionary containing the final results of the model.
        
        Returns:
        str: Formatted string presenting the results.
        """
        formatted_output = f"""
        Model Results:
        ------------------------------
        Best Order of Applying Filters:
        - {' -> '.join(result_dict['best_order'])}
        
        Original Failure Rate: {result_dict['original_failure_rate']}
        Failure Rate After applying Strategy: {result_dict['lowest_failure_rate']}
        
        Production Loss: {result_dict['production_loss']} units
        Relative Improvement in Failure Rate: {result_dict['relative_improvement_percentage']}
        """
        
        return formatted_output
