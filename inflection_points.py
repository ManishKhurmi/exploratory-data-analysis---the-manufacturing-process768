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

    def plot_model_curves(self, predictor_vars, target_var='machine_failure', model='ols', combine_plots=0, ncols=3, standardize=False, plot_derivatives=False):
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
                plt.plot(x_values, predict_data['predicted_values'], label=f'{variable}', color='blue')

                if plot_derivatives:
                    # Plot first and second derivatives
                    y_first_derivative = self.first_derivative(x_values, coefficients)
                    y_second_derivative = self.second_derivative(x_values, coefficients)

                    plt.plot(x_values, y_first_derivative, label=f'{variable} First Derivative', linestyle='--', color='green')
                    plt.plot(x_values, y_second_derivative, label=f'{variable} Second Derivative', linestyle=':', color='red')

                # Add data label to the end of each line
                plt.text(x_values[-1], predict_data['predicted_values'].iloc[-1], f'{variable}', fontsize=9, verticalalignment='center')

            # Customize combined plot
            plt.title(f'{model.upper()} Regression Curves')
            plt.xlabel('Standardized Predictor Variables' if standardize else 'Predictor Variables')
            plt.ylabel(f'Predicted {target_var}')
            plt.grid(True)
            
            # Add the legend
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
                axes[i].plot(x_values, predict_data['predicted_values'], color='blue', label=f'{variable}')

                if plot_derivatives:
                    # Plot first and second derivatives
                    y_first_derivative = self.first_derivative(x_values, coefficients)
                    y_second_derivative = self.second_derivative(x_values, coefficients)

                    axes[i].plot(x_values, y_first_derivative, label=f'{variable} First Derivative', linestyle='--', color='green')
                    axes[i].plot(x_values, y_second_derivative, label=f'{variable} Second Derivative', linestyle=':', color='red')

                axes[i].set_title(f'{model.upper()} Regression Curve for {variable}')
                axes[i].set_xlabel('Standardized ' + variable if standardize else variable)
                axes[i].set_ylabel(f'Predicted {target_var}')
                axes[i].grid(True)

                # Add data label to the end of each line
                axes[i].text(x_values[-1], predict_data['predicted_values'].iloc[-1], f'{variable}', fontsize=9, verticalalignment='center')

            # Add legend to each subplot
            handles, labels = axes[i].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper right', fontsize=9)

            plt.tight_layout()
            plt.show()


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
# # model.plot_model_curves(predictor_vars, model='logit', ncols=3, standardize=False)


# inflection_points = model.calculate_inflection_points(predictor_vars)
# print(inflection_points)

# model.plot_model_curves(predictor_vars, model='logit', ncols=3, standardize=False)

bool_type_L_only = failure_data_df['Type']=='L'
type_L_df = failure_data_df[bool_type_L_only]

type_L_model = Models(type_L_df)

model.plot_model_curves(predictor_vars, model='logit', ncols=3, standardize=True, plot_derivatives=True)
type_L_model.plot_model_curves(predictor_vars, model='logit', ncols=3, standardize=True, plot_derivatives=True)

# TODO: put this in the appendix of the ipynb notebook.