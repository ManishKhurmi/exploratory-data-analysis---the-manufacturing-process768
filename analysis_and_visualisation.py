from manufacturing_eda_classes import LoadData, DataFrameInfo, DataTransform, Plotter, Models
import pandas as pd

# print('#'* 100)
# load_data = LoadData('failure_data.csv')
# failure_data_df = load_data.df
# print(failure_data_df.head(2))

# print('#'* 100)
# info = DataFrameInfo(failure_data_df)
# print(info.column_names())

# print('#'* 100)
# dt = DataTransform(failure_data_df)
# type_dummy_df = dt.create_dummies_from_column('Type')
# print(type_dummy_df.head(2))

# print('#'* 100)
# model = Models(failure_data_df)
# print(model.chi_squared_test_df(binary_cols=['Machine failure', 'RNF']))
##############################################################################################################################

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

# Analyses and Visualisation 

# Task 1: Current operating ranges 
dt = DataTransform(failure_data_df)
machine_failure_col_mapping = {
    'Machine failure': 'machine_failure',
    'Air temperature [K]': 'air_temperature',
    'Process temperature [K]': 'process_temperature',
    'Rotational speed [rpm]': 'rotational_speed_actual',
    'Torque [Nm]': 'torque',
    'Tool wear [min]': 'tool_wear'
}

# rename df 
failure_data_df = dt.rename_colunms(machine_failure_col_mapping)
failure_data_df.head(2)
print(failure_data_df.columns)

columns_of_interest = ['air_temperature', 'process_temperature', 'rotational_speed_actual','torque', 'tool_wear']

def range_df(df, columns):
    min_values = df[columns].min()
    max_values = df[columns].max()

    range_df = pd.DataFrame({
    'Variables': columns,
    'Min': min_values.values,
    'Max': max_values.values
    })

    range_df.set_index('Variables', inplace=True)
    return range_df

print('#'*80)
print('Range of Variables across failure_data')
range_df_failure_data = range_df(failure_data_df, columns=['air_temperature', 'process_temperature', 'rotational_speed_actual','torque', 'tool_wear'])
print(range_df_failure_data)

print('#'*80)
unique_product_types = failure_data_df['Type'].unique()

for i in unique_product_types:
    print(f"\nProduct Type {i}:")
    filtered_df = failure_data_df[failure_data_df['Type'] == i]
    type_range_df = range_df(filtered_df, columns=columns_of_interest)
    print(type_range_df)


##################################################################################################
# The management would also like to know the upper limits of tool wear the machine tools have been operating at. 
# Create a visualisation displaying the number of tools operating at different tool wear values.
    # Histogram?
import seaborn as sns 
import matplotlib.pyplot as plt 

# Maybe pair this with a boxplot 
# plott = Plotter(failure_data_df)
# plott.histplot('tool_wear')
# plt.show()
##################################################################################################
# Task 2: Determine the failure rate in the process 
##################################################################################################
# You've been tasked with determining how many and the leading causes of failure are in the manufacturing process.

# T2a) Determine and visualise how many failures have happened in the process, what percentage is this of the total? 

# T2a) Barplot of machine failures 
failure_types = ['machine_failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']

# Barplot
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
print(f'Do the Total number of other machine failure types (RNF, PWF,..etc) equal the `machine_failure` total?: \n{are_failures_equal}')

if not are_failures_equal:
    print('''This result is `False`, indicating that the `machine_failure` flag can represent 
         multiple failure types for a single observation. For example, one observation can contain both 
         a random failure and a PWF failure.''')


# - **TWF (tool wear failure)**: Failure in the process due to the tool wearing out
# - **head dissipation failure (HDF)**: Lack of heat dissipation caused the process failure
# - **power failure (PWF)**: Failure in the process due to lack of power from the tool to complete the process
# - **overstrain failure (OSF)**: Failure due to the tool overstraining during the process
# - **random failures (RNF)**: Failures in the process which couldn't be categorised
##################################################################################################

# T2a) Calculate the % machine failures 
print('\n')
percentage_failures_df = failure_sum_df / len(failure_data_df) * 100 
print(percentage_failures_df) # TODO: needs a column heading

# total_percentage_failures = percentage_failures_df.sum()
# print(percentage_failures_df)

ax = sns.barplot(percentage_failures_df)
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f') # formating for integer 
plt.show()

# Check if the failures are being caused based on the quality of the product.
    # calculate the % split of Types (L, M, H) across the data set
    # % Machine failures based on Type e.g. 30% are due to Low quality etc (not actuals)


# What seems to be the leading causes of failure in the process? 
    # Create a visualisation of the number of failures due to each possible cause during the manufacturing process.
        # bar chart of the different failure types 

##################################################################################################
# Task 3 
# model = Models(failure_data_df)
# model.logit(formula='RNF ~ air_temperature + ')