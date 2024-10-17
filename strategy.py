import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
import math
from itertools import permutations

from manufacturing_eda_classes import LoadData, DataFrameInfo, DataTransform, Plotter, Models

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
# Setup df & instances for Plots 
failure_data_df = dt.rename_colunms(machine_failure_col_mapping)
model = Models(failure_data_df)
predictor_vars = ['torque', 'rotational_speed_actual', 'air_temperature', 'process_temperature', 'tool_wear']

failure_data_df_copy = failure_data_df.copy()
# 1) Logistic Regression: Model Curves 
model.plot_model_curves(predictor_vars, model='logit', ncols=3, standardize=False)

# 2) Plot of Actuals with Derivatives - Shows that the fist and second derivative don't tell us much as the values are not interpretable with Actuals & logistic regression.
model.plot_model_curves(predictor_vars, model='logit', ncols=3, standardize=False, plot_derivatives=True, local_minima=True)

# 3) Standardised Vars with First and second derivative
dict_minima_coordinates = model.plot_model_curves(predictor_vars, model='logit', ncols=3, standardize=True, plot_derivatives=True, local_minima=True)
print(dict_minima_coordinates)
print('Results \n:')
dict_descaled_minima_coordinates = model.inverse_scale_minima(dict_minima_coordinates)
print(dict_descaled_minima_coordinates)
# Extract the rounded x-values of the second derivatives
theoretical_strategy = model.extract_x_value_of_second_derivative(dict_descaled_minima_coordinates)
# We will call this our Theoretical Strategy 
print(theoretical_strategy)

# Business strategy based on speculation
business_strategy = {
    'torque': [60],
    'rotational_speed_actual': [1900],
    'air_temperature': [304],  
    'process_temperature': [312],  
    'tool_wear': [200]
}

# 4) Plot of Theoretical and Business Strategy on Actuals
model = Models(failure_data_df_copy)
model.plot_model_curves(predictor_vars, model='logit', ncols=3, standardize=False, plot_derivatives=False, local_minima=True, theoretical_strategy=theoretical_strategy, 
                                        business_strategy=business_strategy)

# Results of Strategy
dict_impact_of_theoretical_strategy = model.impact_of_strategy(theoretical_strategy)
result_theoretical_strategy = model.present_results(result_dict=dict_impact_of_theoretical_strategy)

dict_impact_of_business_strategy = model.impact_of_strategy(business_strategy)
result_business_approach = model.present_results(result_dict=dict_impact_of_business_strategy)


print(f'Results of Theoretical Strategy: \n {result_theoretical_strategy}')
print(f'Results of Business Strategy" \n {result_business_approach}')