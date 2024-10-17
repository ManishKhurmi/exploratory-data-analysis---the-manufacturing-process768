import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
import math

from manufacturing_eda_classes import LoadData, DataFrameInfo, DataTransform

# Data Preprocessing
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

    def plot_model_curves(self, predictor_vars, target_var='machine_failure', model='ols', combine_plots=0, 
                        ncols=3, standardize=False, plot_derivatives=False, local_minima=False, 
                        theoretical_strategy=None, business_strategy=None):
        """
        Plot OLS or Logit model regression curves for each predictor variable or combine all into one plot.
        Optionally standardizes the variables and returns the minima coordinates for first and second derivatives.
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

    def inverse_scale_minima(self, minima_coordinates):
        """
        Inverse the scaling applied to the minima coordinates using the data from self.df and the scalers saved in plot_model_curves.

        Parameters:
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
# Setup df & instances 
failure_data_df = dt.rename_colunms(machine_failure_col_mapping)

model = Models(failure_data_df)
predictor_vars = ['torque', 'rotational_speed_actual', 'air_temperature', 'process_temperature', 'tool_wear']

failure_data_df_copy = failure_data_df.copy()
# minima_coords = model.plot_model_curves(predictor_vars, model='logit', ncols=3, standardize=True, plot_derivatives=True, local_minima=True)
# print(minima_coords)

# # # Plot of Actuals with Derivatives - Shows that the fist and second derivative don't tell us much as the values are not interpretable with Actuals & logistic regression.
# model.plot_model_curves(predictor_vars, model='logit', ncols=3, standardize=False, plot_derivatives=True, local_minima=True)

# # Plot of Standardised Vars with first and second derivatives 
model.plot_model_curves(predictor_vars, model='logit', ncols=3, standardize=True, plot_derivatives=True, local_minima=True)

model = Models(failure_data_df_copy)
predictor_vars = ['torque', 'rotational_speed_actual', 'air_temperature', 'process_temperature', 'tool_wear']
# Dictionary with vertical lines for each predictor variable
theoretical_strategy = {
    'torque': [50],
    'rotational_speed_actual': [1750],
    'air_temperature': [300],  # Vertical lines at 15 and 25
    'process_temperature': [310],  # Vertical lines at 70 and 80
    'tool_wear': [190]
}
business_strategy = {
    'torque': [60],
    'rotational_speed_actual': [1900],
    'air_temperature': [304],  # Vertical lines at 15 and 25
    'process_temperature': [312],  # Vertical lines at 70 and 80
    'tool_wear': [200]
}
# Final Strategy
model.plot_model_curves(predictor_vars, model='logit', ncols=3, standardize=False, plot_derivatives=False, local_minima=True, theoretical_strategy=theoretical_strategy, 
                                        business_strategy=business_strategy)



# model = Models(failure_data_df_copy)

minima_coords = model.plot_model_curves(predictor_vars, model='logit', ncols=3, standardize=True, plot_derivatives=True, local_minima=True)
print(minima_coords)
print('\n')
print(type(minima_coords))

result_dict = model.inverse_scale_minima(minima_coords)
print(result_dict)


# minima_coords = {
#     'torque': {'first_derivative_min': (None, None), 'second_derivative_min': (1.302532048803255, -0.09621497698616391)},
#     'rotational_speed_actual': {'first_derivative_min': (None, None), 'second_derivative_min': (1.3150438456951041, -0.09622486847047655)},
#     'air_temperature': {'first_derivative_min': (None, None), 'second_derivative_min': (1.2994355529261714, -0.09621017351230728)},
#     'process_temperature': {'first_derivative_min': (None, None), 'second_derivative_min': (1.2896496029825735, -0.0961887904634136)},
#     'tool_wear': {'first_derivative_min': (None, None), 'second_derivative_min': (1.3044680174931413, -0.09621750346930949)}
# }

# model = Models(failure_data_df)
# Assuming df is your dataframe and contains the predictor variables
# result_dict = model.inverse_scale_minima(minima_coords)
# print(result_dict)


# test_minima_coordinates = {
#     'torque': {'first_derivative_min': (None, None), 'second_derivative_min': (1.02532048803255, -0.09621497698616391)},
#     'rotational_speed_actual': {'first_derivative_min': (None, None), 'second_derivative_min': (1.3150438456951041, -0.09622486847047655)},
#     'air_temperature': {'first_derivative_min': (None, None), 'second_derivative_min': (1.2994355529261714, -0.09621017351230728)},
#     'process_temperature': {'first_derivative_min': (None, None), 'second_derivative_min': (1.2896496029825735, -0.0961887904634136)},
#     'tool_wear': {'first_derivative_min': (None, None), 'second_derivative_min': (1.3044680174931413, -0.09621750346930949)}
# }

# print('Result:\n')
# result_dict = model.inverse_scale_minima(minima_coords)
# print(result_dict)



# descaler = descale(failure_data_df_copy)
# result = descale.unscale_minima_coordinates(minima_coordinates=minima_coords)

# # result = model.unscale_minima_coordinates(minima_coordinates=minima_coords)
# print(result)



# Assuming you already have the DataFrame 'df' and 'minima_coords' to work with


# Call the 'unscale_minima_coordinates' method on the instance, passing the minima_coordinates



expected = {'torque': {'first_derivative_min': (None, None), 'second_derivative_min': (49.71993233139408, -0.09621497698616391)}, 
 'rotational_speed_actual': {'first_derivative_min': (None, None), 'second_derivative_min': (1750.9393939393938, -0.09622486847047655)}, 
 'air_temperature': {'first_derivative_min': (None, None), 'second_derivative_min': (302.5, -0.09621017351230728)}, 
 'process_temperature': {'first_derivative_min': (None, None), 'second_derivative_min': (311.830303030303, -0.0961887904634136)}, 
 'tool_wear': {'first_derivative_min': (None, None), 'second_derivative_min': (189.11111111111111, -0.09621750346930949)}}


# Models & Minima 
# logit_model_machine_failure = model.logit(formula = "machine_failure ~ air_temperature + process_temperature + rotational_speed_actual + torque + tool_wear", model_summary=1)
# ols = model.ols(formula = "machine_failure ~ air_temperature + process_temperature + rotational_speed_actual + torque + tool_wear", model_summary=1)
# # model.plot_model_curves(predictor_vars, model='logit', ncols=3, standardize=False, plot_derivatives=False, local_minima=False )
# minima_coords = model.plot_model_curves(predictor_vars, model='logit', ncols=3, standardize=True, plot_derivatives=True, local_minima=True)
# print('actual minima dict: \n')
# print(minima_coords)

# # Unscaling:
# unscaled_dict = model.unscale_minima_coordinates(minima_coords)
# print('Unscaled Cords:')
# print(unscaled_dict)

# minima_coords = {
#     'torque': {'first_derivative_min': (None, None), 'second_derivative_min': (1.02532048803255, -0.09621497698616391)},
#     'rotational_speed_actual': {'first_derivative_min': (None, None), 'second_derivative_min': (1.3150438456951041, -0.09622486847047655)},
#     'air_temperature': {'first_derivative_min': (None, None), 'second_derivative_min': (1.2994355529261714, -0.09621017351230728)},
#     'process_temperature': {'first_derivative_min': (None, None), 'second_derivative_min': (1.2896496029825735, -0.0961887904634136)},
#     'tool_wear': {'first_derivative_min': (None, None), 'second_derivative_min': (1.3044680174931413, -0.09621750346930949)}
# }

######################################################################################################
# Testing 
# failure_data_df = dt.rename_colunms(machine_failure_col_mapping)
# model = Models(failure_data_df)
# test_minima_coordinates = {
#     'torque': {'first_derivative_min': (None, None), 'second_derivative_min': (1.02532048803255, -0.09621497698616391)},
#     'rotational_speed_actual': {'first_derivative_min': (None, None), 'second_derivative_min': (1.3150438456951041, -0.09622486847047655)},
#     'air_temperature': {'first_derivative_min': (None, None), 'second_derivative_min': (1.2994355529261714, -0.09621017351230728)},
#     'process_temperature': {'first_derivative_min': (None, None), 'second_derivative_min': (1.2896496029825735, -0.0961887904634136)},
#     'tool_wear': {'first_derivative_min': (None, None), 'second_derivative_min': (1.3044680174931413, -0.09621750346930949)}
# }
# unscaled_dict = model.unscale_minima_coordinates(test_minima_coordinates)
# print('Unscaled Cords:')
# print(unscaled_dict)

# Corrently unscaled variables 
# Unscaled Cords:
# expected = {'torque': {'first_derivative_min': (None, None), 'second_derivative_min': (49.71993233139408, -0.09621497698616391)}, 
#  'rotational_speed_actual': {'first_derivative_min': (None, None), 'second_derivative_min': (1750.9393939393938, -0.09622486847047655)}, 
#  'air_temperature': {'first_derivative_min': (None, None), 'second_derivative_min': (302.5, -0.09621017351230728)}, 
#  'process_temperature': {'first_derivative_min': (None, None), 'second_derivative_min': (311.830303030303, -0.0961887904634136)}, 
#  'tool_wear': {'first_derivative_min': (None, None), 'second_derivative_min': (189.11111111111111, -0.09621750346930949)}}



######################################################################################################




