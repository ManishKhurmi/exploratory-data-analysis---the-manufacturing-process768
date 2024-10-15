import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
import math


from manufacturing_eda_classes import LoadData, DataFrameInfo, DataTransform

print('##############################################################################')
print('Step 0: Load the Data')
load_data = LoadData('failure_data.csv')  # Instantiate the class with your file name
failure_data_df = load_data.df  # Access the loaded DataFrame
print('##############################################################################')
print('Step 1: Initial Data Cleaning')
dt = DataTransform(failure_data_df)
failure_data_df = dt.drop_column(['Unnamed: 0', 'Product ID'])

dt = DataTransform(failure_data_df)
type_dummy_df = dt.create_dummies_from_column('Type')
failure_data_df = dt.concat_dataframes(type_dummy_df)

info = DataFrameInfo(failure_data_df)
print(f"\nColumns After concatination: \n{info.column_names()}")
print('##############################################################################')
print('Step 2: Impute missing values')
imputation_dict = {
    'Air temperature [K]': 'median',
    'Process temperature [K]': 'mean',
    'Tool wear [min]': 'median'
}
dt = DataTransform(failure_data_df)
failure_data_df = dt.impute_missing_values(imputation_dict)
info = DataFrameInfo(failure_data_df)
print(f"\nCheck\nPercentage of Null Values for each column after imputation: \n{info.percentage_of_null()}")
print('##############################################################################')
print('Step 3: Treating Skewness')
dt = DataTransform(failure_data_df)
info = DataFrameInfo(failure_data_df)
print(f"\nSkew Test Before treatement: {info.skew_test('Rotational speed [rpm]')}")
failure_data_df = dt.treat_skewness(column_name='Rotational speed [rpm]', normalied_column_name='rotational_speed_normalised', method='yeojohnson') # 
info = DataFrameInfo(failure_data_df)
print(f"\nSkew Test After treatement: {info.skew_test('rotational_speed_normalised')}")
print('##############################################################################')
print('Step 4: Removing Outliers')
dt = DataTransform(failure_data_df)

info = DataFrameInfo(failure_data_df)
outlier_columns = ['rotational_speed_normalised', 'Torque [Nm]', 'Process temperature [K]']
print(f"\nBefore Removing outliers: {info.describe_statistics(outlier_columns)}")

failure_data_df, _, _ = dt.remove_outliers_with_optimiser(outlier_columns, key_ID='UDI', method='IQR', suppress_output=True)

info = DataFrameInfo(failure_data_df)
print(f"\nAfter Removing outliers: {info.describe_statistics(outlier_columns)}")
print('##############################################################################')
print('Step 5: Run Diagnostics')
info = DataFrameInfo(failure_data_df)
print(f"Shape of the DataFrame: {info.return_shape()}")
print(f"Percentage of missing values in each column:\n{info.percentage_of_null()}")
print(f"List of column names in the DataFrame: {info.column_names()}")
print('##############################################################################')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler

class Models: 
    def __init__(self, df):
        self.df = df 

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

    def first_derivative(self, x, coefficients):
        """Calculate the first derivative of the logistic regression curve."""
        # This assumes the logistic function f(x) = 1 / (1 + exp(- (b0 + b1 * x)))
        return np.exp(-x) / (1 + np.exp(-x))**2

    def second_derivative(self, x, coefficients):
        """Calculate the second derivative of the logistic regression curve."""
        # Second derivative for logistic regression
        return -np.exp(-x) * (1 - np.exp(-x)) / (1 + np.exp(-x))**3

    def find_local_minima(self, x_values, y_values):
        """Helper function to find local minima in a given set of x and y values."""
        local_min_x = None
        local_min_y = None
        for i in range(1, len(y_values) - 1):
            if y_values[i - 1] > y_values[i] < y_values[i + 1]:
                local_min_x = x_values[i]
                local_min_y = y_values[i]
                break  # Return the first local minimum found
        return local_min_x, local_min_y
    
    def plot_model_curves(self, predictor_vars, target_var='machine_failure', model='ols', combine_plots=0, ncols=3, standardize=False, plot_derivatives=False, local_minima=False):
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

        # Get the coefficients for the model (used for derivatives)
        coefficients = model_fit.params

        minima_coordinates = {}

        if combine_plots == 1:
            # Combine all plots into a single plot
            plt.figure(figsize=(10, 6))
            for variable in predictor_vars:
                x_values = np.linspace(self.df[variable].min(), self.df[variable].max(), 100)

                # Create a DataFrame for prediction, setting other variables to their mean dynamically
                predict_data = pd.DataFrame({
                    var: [self.df[var].mean()] * len(x_values) for var in predictor_vars
                })
                predict_data[variable] = x_values
                predict_data['predicted_values'] = model_fit.predict(predict_data)

                plt.plot(x_values, predict_data['predicted_values'], label=f'{variable}', color='blue')

                if plot_derivatives or local_minima:
                    y_first_derivative = self.first_derivative(x_values, coefficients)
                    y_second_derivative = self.second_derivative(x_values, coefficients)

                    if plot_derivatives:
                        plt.plot(x_values, y_first_derivative, label=f'{variable} First Derivative', linestyle='--', color='green')
                        plt.plot(x_values, y_second_derivative, label=f'{variable} Second Derivative', linestyle=':', color='red')

                    if local_minima:
                        first_min_x, first_min_y = self.find_local_minima(x_values, y_first_derivative)
                        second_min_x, second_min_y = self.find_local_minima(x_values, y_second_derivative)

                        minima_coordinates[variable] = {
                            'first_derivative_min': (first_min_x, first_min_y),
                            'second_derivative_min': (second_min_x, second_min_y)
                        }

                        # Plot local minima
                        plt.scatter(first_min_x, first_min_y, color='orange', label=f'{variable} First Derivative Min')
                        plt.scatter(second_min_x, second_min_y, color='purple', label=f'{variable} Second Derivative Min')

                plt.text(x_values[-1], predict_data['predicted_values'].iloc[-1], f'{variable}', fontsize=9, verticalalignment='center')

            plt.title(f'{model.upper()} Regression Curves')
            plt.xlabel('Standardized Predictor Variables' if standardize else 'Predictor Variables')
            plt.ylabel(f'Predicted {target_var}')
            plt.grid(True)
            plt.legend(loc='upper left', fontsize=9)
            plt.tight_layout()
            plt.show()

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

                if plot_derivatives or local_minima:
                    y_first_derivative = self.first_derivative(x_values, coefficients)
                    y_second_derivative = self.second_derivative(x_values, coefficients)

                    if plot_derivatives:
                        axes[i].plot(x_values, y_first_derivative, label=f'{variable} First Derivative', linestyle='--', color='green')
                        axes[i].plot(x_values, y_second_derivative, label=f'{variable} Second Derivative', linestyle=':', color='red')

                    if local_minima:
                        first_min_x, first_min_y = self.find_local_minima(x_values, y_first_derivative)
                        second_min_x, second_min_y = self.find_local_minima(x_values, y_second_derivative)

                        minima_coordinates[variable] = {
                            'first_derivative_min': (first_min_x, first_min_y),
                            'second_derivative_min': (second_min_x, second_min_y)
                        }

                        # Plot local minima
                        axes[i].scatter(first_min_x, first_min_y, color='orange', label=f'{variable} First Derivative Min')
                        axes[i].scatter(second_min_x, second_min_y, color='purple', label=f'{variable} Second Derivative Min')

                axes[i].set_title(f'{model.upper()} Regression Curve for {variable}')
                axes[i].set_xlabel('Standardized ' + variable if standardize else variable)
                axes[i].set_ylabel(f'Predicted {target_var}')
                axes[i].grid(True)
                axes[i].text(x_values[-1], predict_data['predicted_values'].iloc[-1], f'{variable}', fontsize=9, verticalalignment='center')

            handles, labels = axes[i].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper right', fontsize=9)
            plt.tight_layout()
            plt.show()

        return minima_coordinates











# Approach with local minima dictionary
    # def plot_model_curves(self, predictor_vars, target_var='machine_failure', model='ols', combine_plots=0, ncols=3, standardize=False, plot_derivatives=False, local_minima=False):
    #     """Plot OLS or Logit model regression curves for each predictor variable or combine all into one plot."""
        
    #     minima_coordinates = {}
    #     unscaled_minima_coordinates = {}

    #     # Optionally standardize the variables
    #     if standardize:
    #         scaler = StandardScaler()
    #         scaled_df = pd.DataFrame(scaler.fit_transform(self.df[predictor_vars]), columns=predictor_vars)
    #     else:
    #         scaled_df = self.df.copy()

    #     # Fit the chosen model: OLS or Logit
    #     formula = f'{target_var} ~ ' + ' + '.join(predictor_vars)
    #     if model == 'ols':
    #         model_fit = self.ols(formula)
    #     elif model == 'logit':
    #         model_fit = self.logit(formula)
    #     else:
    #         raise ValueError("Model must be 'ols' or 'logit'.")

    #     # Get the coefficients for the model (used for derivatives)
    #     coefficients = model_fit.params

    #     if combine_plots == 1:
    #         plt.figure(figsize=(10, 6))
    #         for variable in predictor_vars:
    #             x_values = np.linspace(scaled_df[variable].min(), scaled_df[variable].max(), 100)
    #             unscaled_x_values = np.linspace(self.df[variable].min(), self.df[variable].max(), 100)

    #             predict_data = pd.DataFrame({
    #                 var: [scaled_df[var].mean()] * len(x_values) for var in predictor_vars
    #             })
    #             predict_data[variable] = x_values
    #             predict_data['predicted_values'] = model_fit.predict(predict_data)

    #             plt.plot(x_values, predict_data['predicted_values'], label=f'{variable}', color='blue')

    #             if plot_derivatives or local_minima:
    #                 y_first_derivative = self.first_derivative(x_values, coefficients)
    #                 y_second_derivative = self.second_derivative(x_values, coefficients)

    #                 if plot_derivatives:
    #                     plt.plot(x_values, y_first_derivative, label=f'{variable} First Derivative', linestyle='--', color='green')
    #                     plt.plot(x_values, y_second_derivative, label=f'{variable} Second Derivative', linestyle=':', color='red')

    #                 if local_minima:
    #                     first_min_x, first_min_y = self.find_local_minima(x_values, y_first_derivative)
    #                     second_min_x, second_min_y = self.find_local_minima(x_values, y_second_derivative)

    #                     minima_coordinates[variable] = {
    #                         'first_derivative_min': (first_min_x, first_min_y),
    #                         'second_derivative_min': (second_min_x, second_min_y)
    #                     }

    #                     # Get the unscaled minima by reverse transforming the scaled minima
    #                     if standardize:
    #                         unscaled_first_min_x = scaler.inverse_transform([[first_min_x] + [0]*(len(predictor_vars)-1)])[0][0]
    #                         unscaled_second_min_x = scaler.inverse_transform([[second_min_x] + [0]*(len(predictor_vars)-1)])[0][0]
    #                     else:
    #                         unscaled_first_min_x = first_min_x
    #                         unscaled_second_min_x = second_min_x

    #                     unscaled_minima_coordinates[variable] = {
    #                         'first_derivative_min': (unscaled_first_min_x, first_min_y),
    #                         'second_derivative_min': (unscaled_second_min_x, second_min_y)
    #                     }

    #                     # Plot local minima
    #                     plt.scatter(first_min_x, first_min_y, color='orange', label=f'{variable} First Derivative Min')
    #                     plt.scatter(second_min_x, second_min_y, color='purple', label=f'{variable} Second Derivative Min')

    #             plt.text(x_values[-1], predict_data['predicted_values'].iloc[-1], f'{variable}', fontsize=9, verticalalignment='center')

    #         plt.title(f'{model.upper()} Regression Curves')
    #         plt.xlabel('Standardized Predictor Variables' if standardize else 'Predictor Variables')
    #         plt.ylabel(f'Predicted {target_var}')
    #         plt.grid(True)
    #         plt.legend(loc='upper left', fontsize=9)
    #         plt.tight_layout()
    #         plt.show()

    #     else:
    #         n_plots = len(predictor_vars)
    #         nrows = math.ceil(n_plots / ncols)
    #         fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 6 * nrows))
    #         axes = axes.flatten()

    #         for i, variable in enumerate(predictor_vars):
    #             x_values = np.linspace(scaled_df[variable].min(), scaled_df[variable].max(), 100)
    #             unscaled_x_values = np.linspace(self.df[variable].min(), self.df[variable].max(), 100)

    #             predict_data = pd.DataFrame({
    #                 var: [scaled_df[var].mean()] * len(x_values) for var in predictor_vars
    #             })
    #             predict_data[variable] = x_values
    #             predict_data['predicted_values'] = model_fit.predict(predict_data)

    #             axes[i].plot(x_values, predict_data['predicted_values'], color='blue', label=f'{variable}')

    #             if plot_derivatives or local_minima:
    #                 y_first_derivative = self.first_derivative(x_values, coefficients)
    #                 y_second_derivative = self.second_derivative(x_values, coefficients)

    #                 if plot_derivatives:
    #                     axes[i].plot(x_values, y_first_derivative, label=f'{variable} First Derivative', linestyle='--', color='green')
    #                     axes[i].plot(x_values, y_second_derivative, label=f'{variable} Second Derivative', linestyle=':', color='red')

    #                 if local_minima:
    #                     first_min_x, first_min_y = self.find_local_minima(x_values, y_first_derivative)
    #                     second_min_x, second_min_y = self.find_local_minima(x_values, y_second_derivative)

    #                     minima_coordinates[variable] = {
    #                         'first_derivative_min': (first_min_x, first_min_y),
    #                         'second_derivative_min': (second_min_x, second_min_y)
    #                     }

    #                     # Get the unscaled minima by reverse transforming the scaled minima
    #                     if standardize:
    #                         unscaled_first_min_x = scaler.inverse_transform([[first_min_x] + [0]*(len(predictor_vars)-1)])[0][0]
    #                         unscaled_second_min_x = scaler.inverse_transform([[second_min_x] + [0]*(len(predictor_vars)-1)])[0][0]
    #                     else:
    #                         unscaled_first_min_x = first_min_x
    #                         unscaled_second_min_x = second_min_x

    #                     unscaled_minima_coordinates[variable] = {
    #                         'first_derivative_min': (unscaled_first_min_x, first_min_y),
    #                         'second_derivative_min': (unscaled_second_min_x, second_min_y)
    #                     }

    #                     # Plot local minima
    #                     axes[i].scatter(first_min_x, first_min_y, color='orange', label=f'{variable} First Derivative Min')
    #                     axes[i].scatter(second_min_x, second_min_y, color='purple', label=f'{variable} Second Derivative Min')

    #             axes[i].set_title(f'{model.upper()} Regression Curve for {variable}')
    #             axes[i].set_xlabel('Standardized ' + variable if standardize else variable)
    #             axes[i].set_ylabel(f'Predicted {target_var}')
    #             axes[i].grid(True)
    #             axes[i].text(x_values[-1], predict_data['predicted_values'].iloc[-1], f'{variable}', fontsize=9, verticalalignment='center')

    #         handles, labels = axes[i].get_legend_handles_labels()
    #         fig.legend(handles, labels, loc='upper right', fontsize=9)
    #         plt.tight_layout()
    #         plt.show()

    #     return minima_coordinates, unscaled_minima_coordinates




    # ORG
    # def plot_model_curves(self, predictor_vars, target_var='machine_failure', model='ols', combine_plots=0, ncols=3, standardize=False, plot_derivatives=False):
    #     """Plot OLS or Logit model regression curves for each predictor variable or combine all into one plot."""
        
    #     # Optionally standardize the variables
    #     if standardize:
    #         scaler = StandardScaler()
    #         self.df[predictor_vars] = scaler.fit_transform(self.df[predictor_vars])

    #     # Fit the chosen model: OLS or Logit
    #     formula = f'{target_var} ~ ' + ' + '.join(predictor_vars)
    #     if model == 'ols':
    #         model_fit = self.ols(formula)
    #     elif model == 'logit':
    #         model_fit = self.logit(formula)
    #     else:
    #         raise ValueError("Model must be 'ols' or 'logit'.")

    #     # Get the coefficients for the model (used for derivatives)
    #     coefficients = model_fit.params

    #     if combine_plots == 1:
    #         # Combine all plots into a single plot
    #         plt.figure(figsize=(10, 6))
    #         for variable in predictor_vars:
    #             # Generate a sequence of values over the range of the selected variable
    #             x_values = np.linspace(self.df[variable].min(), self.df[variable].max(), 100)

    #             # Create a DataFrame for prediction, setting other variables to their mean dynamically
    #             predict_data = pd.DataFrame({
    #                 var: [self.df[var].mean()] * len(x_values) for var in predictor_vars
    #             })

    #             # Ensure that only the specific variable changes
    #             predict_data[variable] = x_values

    #             # Predict the target variable using the chosen model
    #             predict_data['predicted_values'] = model_fit.predict(predict_data)

    #             # Plot the regression/logit curve on the same plot
    #             plt.plot(x_values, predict_data['predicted_values'], label=f'{variable}', color='blue')

    #             if plot_derivatives:
    #                 # Plot first and second derivatives
    #                 y_first_derivative = self.first_derivative(x_values, coefficients)
    #                 y_second_derivative = self.second_derivative(x_values, coefficients)

    #                 plt.plot(x_values, y_first_derivative, label=f'{variable} First Derivative', linestyle='--', color='green')
    #                 plt.plot(x_values, y_second_derivative, label=f'{variable} Second Derivative', linestyle=':', color='red')

    #             # Add data label to the end of each line
    #             plt.text(x_values[-1], predict_data['predicted_values'].iloc[-1], f'{variable}', fontsize=9, verticalalignment='center')

    #         # Customize combined plot
    #         plt.title(f'{model.upper()} Regression Curves')
    #         plt.xlabel('Standardized Predictor Variables' if standardize else 'Predictor Variables')
    #         plt.ylabel(f'Predicted {target_var}')
    #         plt.grid(True)
            
    #         # Add the legend
    #         plt.legend(loc='upper left', fontsize=9)  # Adjust legend for clarity

    #         plt.tight_layout()
    #         plt.show()

    #     else:
    #         # Calculate number of plots and dynamic layout for individual plots
    #         n_plots = len(predictor_vars)
    #         nrows = math.ceil(n_plots / ncols)  # Dynamically determine the number of rows

    #         # Create subplots with dynamic layout
    #         fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 6 * nrows))
    #         axes = axes.flatten()  # Flatten the axes array for easy iteration

    #         for i, variable in enumerate(predictor_vars):
    #             # Generate a sequence of values over the range of the selected variable
    #             x_values = np.linspace(self.df[variable].min(), self.df[variable].max(), 100)

    #             # Create a DataFrame for prediction, setting other variables to their mean dynamically
    #             predict_data = pd.DataFrame({
    #                 var: [self.df[var].mean()] * len(x_values) for var in predictor_vars
    #             })

    #             # Ensure that only the specific variable changes
    #             predict_data[variable] = x_values

    #             # Predict the target variable using the chosen model
    #             predict_data['predicted_values'] = model_fit.predict(predict_data)

    #             # Plot the OLS or Logit regression curve in the subplot
    #             axes[i].plot(x_values, predict_data['predicted_values'], color='blue', label=f'{variable}')

    #             if plot_derivatives:
    #                 # Plot first and second derivatives
    #                 y_first_derivative = self.first_derivative(x_values, coefficients)
    #                 y_second_derivative = self.second_derivative(x_values, coefficients)

    #                 axes[i].plot(x_values, y_first_derivative, label=f'{variable} First Derivative', linestyle='--', color='green')
    #                 axes[i].plot(x_values, y_second_derivative, label=f'{variable} Second Derivative', linestyle=':', color='red')

    #             axes[i].set_title(f'{model.upper()} Regression Curve for {variable}')
    #             axes[i].set_xlabel('Standardized ' + variable if standardize else variable)
    #             axes[i].set_ylabel(f'Predicted {target_var}')
    #             axes[i].grid(True)

    #             # Add data label to the end of each line
    #             axes[i].text(x_values[-1], predict_data['predicted_values'].iloc[-1], f'{variable}', fontsize=9, verticalalignment='center')

    #         # Add legend to each subplot
    #         handles, labels = axes[i].get_legend_handles_labels()
    #         fig.legend(handles, labels, loc='upper right', fontsize=9)

    #         plt.tight_layout()
    #         plt.show()

    # def plot_derivatives_separately(self, predictor_vars, model='logit', combine_plots=False):
    #     """Plot first and second derivatives separately or combined for each predictor variable."""
        
    #     # Fit the chosen model
    #     formula = f'machine_failure ~ ' + ' + '.join(predictor_vars)
    #     if model == 'logit':
    #         model_fit = self.logit(formula)
    #     else:
    #         raise ValueError("Model must be 'logit' for derivatives.")
        
    #     # Get coefficients for derivatives
    #     coefficients = model_fit.params

    #     # Generate plots for first and second derivatives
    #     for variable in predictor_vars:
    #         x_values = np.linspace(self.df[variable].min(), self.df[variable].max(), 100)
    #         first_deriv = self.first_derivative(x_values, coefficients)
    #         second_deriv = self.second_derivative(x_values, coefficients)

    #         if combine_plots:
    #             # Combine first and second derivatives into one plot
    #             plt.figure(figsize=(10, 5))
    #             plt.plot(x_values, first_deriv, label=f'{variable} First Derivative', color='green')
    #             plt.plot(x_values, second_deriv, label=f'{variable} Second Derivative', color='red', linestyle='--')
    #             plt.title(f'First and Second Derivatives of {variable}')
    #             plt.xlabel(variable)
    #             plt.ylabel('Derivative Values')
    #             plt.grid(True)
    #             plt.legend()
    #             plt.show()

    #         else:
    #             # Plot first derivative
    #             plt.figure(figsize=(10, 5))
    #             plt.plot(x_values, first_deriv, label=f'{variable} First Derivative', color='green')
    #             plt.title(f'First Derivative of {variable}')
    #             plt.xlabel(variable)
    #             plt.ylabel('First Derivative')
    #             plt.grid(True)
    #             plt.legend()
    #             plt.show()

    #             # Plot second derivative
    #             plt.figure(figsize=(10, 5))
    #             plt.plot(x_values, second_deriv, label=f'{variable} Second Derivative', color='red')
    #             plt.title(f'Second Derivative of {variable}')
    #             plt.xlabel(variable)
    #             plt.ylabel('Second Derivative')
    #             plt.grid(True)
    #             plt.legend()
    #             plt.show()


dt = DataTransform(failure_data_df)
machine_failure_col_mapping = {
    'Machine failure': 'machine_failure',
    'Air temperature [K]': 'air_temperature',
    'Process temperature [K]': 'process_temperature',
    'Rotational speed [rpm]': 'rotational_speed_actual',
    'Torque [Nm]': 'torque',
    'Tool wear [min]': 'tool_wear',
    'TWF': 'tool_wear_failure',
    'HDF': 'head_dissapation_failure',
    'PWF': 'power_failure',
    'OSF': 'overstrain_failure',
    'RNF': 'random_failure'
}
failure_data_df = dt.rename_colunms(machine_failure_col_mapping)

model = Models(failure_data_df)
predictor_vars = ['torque', 'rotational_speed_actual', 'air_temperature', 'process_temperature', 'tool_wear']
# logit_model_machine_failure = model.logit(formula = "machine_failure ~ air_temperature + process_temperature + rotational_speed_actual + torque + tool_wear", model_summary=1)

# # model.plot_model_curves(['torque'], model='logit', ncols=3, standardize=True, plot_derivatives=True, local_minima=False )
minima_coords = model.plot_model_curves(predictor_vars, model='logit', ncols=3, standardize=True, plot_derivatives=True, local_minima=True)

# print(minima_coords)
# model.plot_model_curves(
#     predictor_vars=['torque', 'rotational_speed_actual', 'air_temperature', 'process_temperature', 'tool_wear'],
#     target_var='machine_failure',
#     model='logit',               # Use Logit model
#     combine_plots=0,             # Combine all plots into a single plot
#     standardize=False,            # Standardize the predictor variables
#     plot_derivatives=False,       # Plot the first and second derivatives
#     local_minima=True            # Identify and plot the local minima
# )


# # Call plot_model_curves function, passing in predictor variables
# minima_coordinates = model.plot_model_curves(
#     predictor_vars=['torque'],
#     target_var='machine_failure',
#     model='logit',               # Use Logit model
#     combine_plots=0,             # Combine all plots into a single plot
#     standardize=True,            # Standardize the predictor variables
#     plot_derivatives=True,       # Plot the first and second derivatives
#     local_minima=True            # Identify and plot the local minima
# )

# # Output the minima coordinates for both scaled and unscaled values
# print("Minima Coordinates (Standardized):")
# print(minima_coordinates)

# print("\nUnscaled Minima Coordinates:")
# print(unscaled_minima_coordinates)


# Unscaled Minima Coordinates:
# {'rotational_speed_actual': {'first_derivative_min': (nan, None), 'second_derivative_min': (1750.9393939393938, -0.09622486847047655)}}
# Minima Coordinates (Standardized):
# {'torque': {'first_derivative_min': (None, None), 'second_derivative_min': (1.302532048803255, -0.09621497698616391)}}
# Unscaled Minima Coordinates:
# {'torque': {'first_derivative_min': (nan, None), 'second_derivative_min': (52.372727272727275, -0.09621497698616391)}}

# model.plot_model_curves(predictor_vars, model='logit', ncols=3, standardize=False, plot_derivatives=True, local_minima=True)
# model.plot_model_curves(predictor_vars, model='logit', ncols=3, standardize=True, plot_derivatives=True, local_minima=True )

# model.plot_model_curves(predictor_vars, model='logit', standardize=True)
# minima_coords, unscaled_minima_coords = model.plot_model_curves(
#     predictor_vars=['torque', 'air_temperature'], 
#     target_var='machine_failure', 
#     model='logit', 
#     combine_plots=0, 
#     ncols=2, 
#     standardize=True, 
#     plot_derivatives=True, 
#     local_minima=True
# )

# # Access the minima coordinates
# # print(minima_coords)            # Minima in standardized data
# print(unscaled_minima_coords)    # Minima in original scale

#{'torque': {'first_derivative_min': (nan, None), 'second_derivative_min': (52.372727272727275, -0.09621497698616391)}, 'air_temperature': {'first_derivative_min': (nan, None), 'second_derivative_min': (52.343095144197896, -0.09621017351230728)}}


# ols_model_machine_failure = model.ols(formula = "machine_failure ~ air_temperature + process_temperature + rotational_speed_actual + torque + tool_wear", model_summary=1)
# model.plot_model_curves(predictor_vars, model='logit', ncols=3, standardize=False)
# model.plot_model_curves(predictor_vars, model='logit', ncols=3, standardize=False)

################################################################################################################
# Plots separated by prouct type 
# bool_type_L_only = failure_data_df['Type']=='L'
# type_L_df = failure_data_df[bool_type_L_only]
# type_L_model = Models(type_L_df)
# model.plot_model_curves(predictor_vars, model='logit', ncols=3, standardize=True, plot_derivatives=True)
# type_L_model.plot_model_curves(predictor_vars, model='logit', ncols=3, standardize=True, plot_derivatives=True)


# bool_type_M_only = failure_data_df['Type']=='M'
# type_M_df = failure_data_df[bool_type_M_only]
# type_M_model = Models(type_M_df)
# model.plot_model_curves(predictor_vars, model='logit', ncols=3, standardize=True, plot_derivatives=True)
# type_M_model.plot_model_curves(predictor_vars, model='logit', ncols=3, standardize=True, plot_derivatives=True)

# bool_type_H_only = failure_data_df['Type']=='H'
# type_H_df = failure_data_df[bool_type_H_only]
# type_H_model = Models(type_H_df)
# model.plot_model_curves(predictor_vars, target_var='head_dissapation_failure',model='logit', ncols=3, standardize=True, plot_derivatives=True)
# type_H_model.plot_model_curves(predictor_vars, 'head_dissapation_failure', model='logit', ncols=3, standardize=True, plot_derivatives=True)


# type_L_model.plot_model_curves(predictor_vars, model='logit', ncols=3, standardize=False, plot_derivatives=True)

# Plot of First & Second Derivative
# model.plot_model_curves(predictor_vars, model='logit', standardize=True, plot_derivatives=True)
# model.plot_derivatives_separately(predictor_vars=['tool_wear'], model='logit', combine_plots=1)


# model.plot_model_curves(predictor_vars, model='logit', standardize=False, plot_derivatives=False)

# TODO: put this in the appendix of the ipynb notebook.