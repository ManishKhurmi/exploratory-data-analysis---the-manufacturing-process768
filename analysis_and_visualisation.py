from manufacturing_eda_classes import LoadData, DataFrameInfo, DataTransform, Plotter, Models
import pandas as pd
pd.set_option('display.max_columns', None)

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
# TODO: consider doing at the beginning of preprocessing 
# Task 1: Current operating ranges 
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

# - **TWF (tool wear failure)**: Failure in the process due to the tool wearing out
# - **head dissipation failure (HDF)**: Lack of heat dissipation caused the process failure
# - **power failure (PWF)**: Failure in the process due to lack of power from the tool to complete the process
# - **overstrain failure (OSF)**: Failure due to the tool overstraining during the process
# - **random failures (RNF)**: Failures in the process which couldn't be categorised
##################################################################################################
#T2a) Determine and visualise how many failures have happened in the process

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

print('Number of Failures is defined as observation that failured for ANY failure type')
number_of_failures = failure_data_df['failure'].sum()
print(f"Number of Failures: {number_of_failures}")

percentage_failure_rate = (number_of_failures/ len(failure_data_df)) * 100
print(f"Percentage Failure Rate: {percentage_failure_rate}")

ax = sns.countplot(failure_data_df, x='failure', color='red')
# Add data labels to each bar
ax.bar_label(ax.containers[0])
plt.title('Number of Failures (0 vs 1)')
plt.xlabel('Failure')
plt.ylabel('Count')
# Show the plot
plt.show()

# # T2a) Calculate the % failure for each failure type
print('\n')
percentage_failures_df = failure_sum_df / len(failure_data_df) * 100 
print(percentage_failures_df) # TODO: needs a column heading

# Barplot of Percentage Failures
ax = sns.barplot(percentage_failures_df, color='red')
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f%%') 
ax.set_title(' Failure Rate (%) per Failure Type')
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

print('Leading cause of failure is product type L, overstrain_failure and head_dissapation_failures are the highest across the produ')
##################################################################################################
# Scratch work
# Using Group By (not needed)
# Count Plot Count of Failure Types split by Product Type
# group_by_product_type_df = failure_data_df.groupby('Type')
# print('\n')
# print(f"Number of Groups: {len(group_by_product_type_df)}")
# check if this is what is expected 
# for group_name, grouped_df in group_by_product_type_df:
#     print(f"Group: {group_name}")
#     print(f"Length of df: {len(grouped_df)}")
#     print("\n")

# X-axis: Failure Types y = Count of Failures split by Type
# Group the failures by 'Type' and sum the failure columns 
# failures_per_type = failure_data_df.groupby('Type')[['tool_wear_failure', 'head_dissapation_failure', 'power_failure', 'overstrain_failure', 'random_failure']].sum()
# # Transpose the DataFrame to swap rows and columns
# failures_per_type = failures_per_type.T
# # Reorder the columns to be 'L', 'M', 'H'
# failures_per_type = failures_per_type[['L', 'M', 'H']]
# # Failures per Failure Type Split by Product Type
# failures_per_type.plot(kind='bar', figsize=(10,6))
# plt.title('Failures per Failure Type Split by Product Type')
# plt.xticks(rotation=45)
# plt.ylabel('Count of Failures')
# plt.legend(title='Product Type')
# plt.show()

# print(failure_data_df.head(3))
# Count Plot: Number of Machine Failures in out data set
# sns.countplot(data=failure_data_df, x='machine_failure')
# plt.show()
# print(failure_data_df.columns)
# print(failure_data_df.head(1))



#
# What seems to be the leading causes of failure in the process? 
    # Create a visualisation of the number of failures due to each possible cause during the manufacturing process.
        # bar chart of the different failure types 

##################################################################################################
# Task 3 
# model = Models(failure_data_df)
# model.logit(formula='RNF ~ air_temperature + ')