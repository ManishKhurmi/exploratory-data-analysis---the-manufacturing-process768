# Import custom Classes 
from manufacturing_eda_classes import LoadData, DataFrameInfo, DataTransform, Plotter, Models

# Imports 
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
import seaborn as sns

##############################################################################################################################
#####################################      Analysis & Visualisation      #####################################################
##############################################################################################################################
''' 
Introduction:
Mile Stone 4 - Now that the data has been transformed, management would like to draw deeper insights from the data. 
You will dive deeper into the dataset to identify any patterns or trends not visible by your previous analysis. 
By gaining these deeper insights, management can make more informed decisions about the manufacturing process to help to reduce manufacturing failures

Task 1: Current operating ranges 
Task 2: Determine the failure rate in the process 
    Task 2a) Determine and visualise how many failures have happened in the process, what percentage is this of the total? 
'''

# 

# Data Preprocessing - DataTransformation decisions from Milestone 3
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
# rename df 
failure_data_df = dt.rename_colunms(machine_failure_col_mapping)
print('##############################################################################')

###################################################################################################
## Task 1: Current operating ranges  
###################################################################################################

columns_of_interest=['air_temperature', 'process_temperature', 'rotational_speed_actual','torque', 'tool_wear']

info = DataFrameInfo(failure_data_df)
range_df_failure_data = info.range_df(columns_of_interest)

unique_product_types = failure_data_df['Type'].unique()
for i in unique_product_types:
    print(f"\nProduct Type {i}:")
    filtered_df = failure_data_df[failure_data_df['Type'] == i]
    info=DataFrameInfo(filtered_df)
    type_range_df = info.range_df(columns_of_interest)
    print(type_range_df)

# The management would also like to know the upper limits of tool wear the machine tools have been operating at. 
# Create a visualisation displaying the number of tools operating at different tool wear values.
    # Histogram?

# Maybe pair this with a boxplot 
plott = Plotter(failure_data_df)
plott.histplot('tool_wear')
plt.show()
# ##################################################################################################
# # Task 2: Determine the failure rate in the process 
'''
You've been tasked with determining how many and the leading causes of failure are in the manufacturing process.

2a) Determine and visualise how many failures have happened in the process
2b) what percentage is this of the total? - Plot of % Failures
2c) Check if the failures are being caused based on the quality of the product. - Plot of Failures per product 

2d) What seems to be the leading causes of failure in the process? 
2e) Create a visualisation of the number of failures due to each possible cause during the manufacturing process.
'''
# ##################################################################################################
'''2a) Determine and visualise how many failures have happened in the process - Barplot of machine failures'''
# failure_types = ['machine_failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
failure_types = ['machine_failure', 'tool_wear_failure', 'head_dissapation_failure', 'power_failure','overstrain_failure','random_failure']

# Barplot: Number of Failures per Failure Type
# ax = sns.barplot(failure_data_df[failure_types], estimator = sum, ci=None) 
# for container in ax.containers:
#     ax.bar_label(container, fmt='%.0f') # formating for integer 
# plt.show()

print('\n')
failure_sum_df = failure_data_df[failure_types].sum()
print(failure_sum_df) # TODO: needs a column heading

machine_failure_total = failure_sum_df[0]
sum_of_other_machine_failures = sum(failure_sum_df[1:])

# Check if the total of other machine failure types equals the machine_failure count
are_failures_equal = machine_failure_total == sum_of_other_machine_failures

# check if the other machine failure types equal the machine_failure count
print(f'Do the Total number of other failure types (RNF, PWF,..etc) equal the `machine_failure` total?: \n{are_failures_equal}')

if not are_failures_equal:
    print('''This result is `False`, indicating that the `machine_failure` flag can represent 
         multiple failure types for a single observation. For example, one observation can contain both 
         a random failure and a PWF failure.''')

print('#' * 100)
# print(failure_data_df[failure_types].head())

# Create a new 'failure' column using the bitwise OR operator to check for any failure
failure_data_df['failure'] = (
    (failure_data_df['machine_failure'] == 1) | 
    (failure_data_df['tool_wear_failure'] == 1) | 
    (failure_data_df['head_dissapation_failure'] == 1) | 
    (failure_data_df['power_failure'] == 1) | 
    (failure_data_df['overstrain_failure'] == 1) | 
    (failure_data_df['random_failure'] == 1)
).astype(int)

print('Number of Failures is defined as observation that failed for ANY failure type')
number_of_failures = failure_data_df['failure'].sum()
print(f"Number of Failures: {number_of_failures}")

percentage_failure_rate = (number_of_failures/ len(failure_data_df)) * 100
print(f"Percentage Failure Rate: {percentage_failure_rate}")

ax = sns.countplot(failure_data_df, x='failure', color='red')
# Add data labels to each bar
ax.bar_label(ax.containers[0])
plt.title('Number of Failures (a failure is defined as observation that failed for ANY failure type)')
plt.xlabel('Failure')
plt.ylabel('Count')
# Show the plot
plt.show()

'''2b) what percentage is this of the total? - Plot of % Failures'''
# Calculate the % failure for each failure type
print('\n')
percentage_failures_df = failure_sum_df / len(failure_data_df) * 100 
print(percentage_failures_df) # TODO: needs a column heading

# Barplot of Percentage Failures
ax = sns.barplot(percentage_failures_df, color='red')
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f%%') 
ax.set_title('(%) Failure Rate per Failure Type')
ax.set_ylabel('Percentage (%) ')
ax.set_ylim(0, percentage_failures_df.max() * 1.2) 
ax.set_xlabel('Failure Types')
# Rotate x-axis lables
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
plt.show()

# # Check if the failures are being caused based on the quality of the product.
#     # calculate the % split of Types (L, M, H) across the data set
#     # % Machine failures based on Type e.g. 30% are due to Low quality etc (not actuals)

# Countplot: Product Types
ax = sns.countplot(failure_data_df, x='Type', palette='Blues')
for container in ax.containers:
    ax.bar_label(container, fmt='%.0f') # integer formatting
ax.set_title('Count of Product Types')
plt.show()


# Barplot: Failure Types split by Product Type
# Group by 'Type' and sum the failure columns
failures_per_type = failure_data_df.groupby('Type')[['machine_failure','tool_wear_failure', 'head_dissapation_failure', 'power_failure', 'overstrain_failure', 'random_failure']].sum()
# Reset Index
failures_per_type = failures_per_type.reset_index()
# Ensure the 'Type' column is categorical and ordered as 'L', 'M', 'H'
failures_per_type['Type'] = pd.Categorical(failures_per_type['Type'], categories=['L', 'M', 'H'], ordered=True)
# Reindex to ensure order
failures_per_type = failures_per_type.set_index('Type').loc[['L', 'M', 'H']]
# Convert the DataFrame back to a long format using melt
failures_per_type_melted = failures_per_type.reset_index().melt(id_vars='Type', var_name='Failure Type', value_name='Count')
# Define custom colors: different shades of red for 'L', 'M', 'H'
custom_palette = {
    'L': '#d73027',  # Dark red
    'M': '#f46d43',  # Medium red
    'H': '#fdae61'   # Light red
}
# Plot using seaborn's barplot with custom colors based on the product type (L, M, H)
plt.figure(figsize=(10, 6))
sns.barplot(x='Failure Type', y='Count', hue='Type', data=failures_per_type_melted, palette=custom_palette)
# Add labels and title
plt.title('Failures per Product Type')
plt.xlabel('Failure Type')
plt.ylabel('Count of Failures')
plt.legend(title='Product Type')
# Show the plot
plt.xticks(rotation=45)
plt.show()

print('Leading cause of failure is in product type L, although this is biased as there are lot more observations for L product type than the rest.')
print('machine_failure, overstrain_failure and head_dissapation_failures are the highest across the product')

# ##################################################################################################
# # Task 3: A Deeper Understanding of Failures 
# ##################################################################################################
# With the failures identified you will need to dive deeper into what the possible causes of failure might be in the process.
# 3a) For each different possible type of failure try to investigate if there is any correlation between any of the settings the machine was running at. 
from manufacturing_eda_classes import LoadData, DataFrameInfo, DataTransform, Plotter, Models

print(failure_data_df.head())
print(failure_data_df.columns)
dt = DataTransform(failure_data_df)
failure_data_df = dt.drop_column(['UDI', 'Type'])

# Initialize the Plotter with DataFrame
plott = Plotter(failure_data_df)
# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(24, 12))
# Plot the first heatmap on the left subplot (no threshold)
plott.correlation_heatmap(figsize=(10, 8), ax=axes[0])
# Plot the second heatmap with a threshold on the right subplot
plott.correlation_heatmap(threshold=0.75, figsize=(10, 8), ax=axes[1])
# # Adjust layout and show the plots
plt.tight_layout()
plt.show()

plott = Plotter(failure_data_df)
plott.correlation_heatmap()
plt.show()


plott = Plotter(failure_data_df)
plott.correlation_heatmap(threshold=0.40)
plt.show()

print('Machine failure is weakly correlated with OSF, HDF and TWF')
####################################################################################################################################################################################
# 3b) Do the failures happen at certain torque ranges, processing temperatures or rpm?
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
    'tool_wear': [240]
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

print('\n')
print(f'Results of Theoretical Strategy: \n {result_theoretical_strategy}')
print(f'Results of Business Strategy \n {result_business_approach}')

# Suggestions to the Business: Focus on 'Tool Wear' 
####################################################################################################################################################################################
# TODO - Put in Appendix 
# Task 3 - Experimenting with combinations of `Type` & different 'failures'

# model = Models(failure_data_df)
# predictor_vars = ['torque', 'rotational_speed_actual', 'air_temperature', 'process_temperature', 'tool_wear']

# # Prove that LOGIT model is better than OLS based on OLS's r-squared alone and the benefit that logistiric regression provides when variables are binary (1,0).
# logit_model_machine_failure = model.logit(formula = "machine_failure ~ air_temperature + process_temperature + rotational_speed_actual + torque + tool_wear", model_summary=1)
# ols_model_machine_failure = model.ols(formula = "machine_failure ~ air_temperature + process_temperature + rotational_speed_actual + torque + tool_wear", model_summary=1)

# # model.plot_model_curves(predictor_vars, model='logit', combine_plots=1, standardize=True)
# model.plot_model_curves(predictor_vars, model='logit', ncols=3, standardize=False)

# # 3c) Try to identify these risk factors so that the company can make more informed decisions about what settings to run the machines at. 
# Experimenting with filtered datasets: filtering by `Type` and other types of failires e.g. L & HDF....
# dt = DataTransform(failure_data_df)
# model = Models(failure_data_df)
# predictor_vars = ['torque', 'rotational_speed_actual', 'air_temperature', 'process_temperature', 'tool_wear']

# # Filtered For Type `L` products only.
# bool_type_L_only = failure_data_df['Type']=='L'
# type_L_df = failure_data_df[bool_type_L_only]
# type_L_model = Models(type_L_df)

# # Type `L` & y = `OSF`
# type_L_model.plot_model_curves(predictor_vars, target_var='overstrain_failure', model='logit', ncols=3, standardize=False) # **L & OSF** #TODO: find the inflection points here 

# # Type `L` & y = `HDF`
# type_L_model.plot_model_curves(predictor_vars, target_var='head_dissapation_failure', model='logit', ncols=3, standardize=False) # **L & HDF** # #TODO find the inflection points here
# type_L_model.plot_model_curves(predictor_vars, target_var='random_failure', model='logit', ncols=3, standardize=False) # L & RNF
# type_L_model.plot_model_curves(predictor_vars, target_var='machine_failure', model='logit', ncols=3, standardize=False) # L & machine failure 