from manufacturing_eda_classes import LoadData, DataFrameInfo, DataTransform, Plotter, Models

print('#'* 100)
load_data = LoadData('failure_data.csv')
failure_data_df = load_data.df
print(failure_data_df.head(2))

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
info = DataFrameInfo(failure_data_df)
dt = DataTransform(failure_data_df)
print('##############################################################################')

print(failure_data_df.columns)
plott = Plotter(failure_data_df)
plott.histogram_and_skew_sub_plots(variable_list=['Rotational speed [rpm]', 'rotational_speed_normalised'], num_cols = 2)




# plott.histplot('rotational_speed_normalised')
# plott.histogram_and_skew_sub_plots(variable_list=['Rotational speed [rpm]', 'rotational_speed_normalised'], num_cols = 2)


# # Impute NULL values of Air temperature 
# dt = DataTransform(failure_data_df)
# # failure_data_df['Air temperature [K]'] = dt.impute_missing_values(column_name='Air temperature [K]', method='median')

# imputation_dict = {'Air temperature [K]': 'median'}
# # failure_data_df['Air temperature [K]'] = dt.impute_missing_values(imputation_dict)
# air_imputed_df = dt.impute_missing_values(imputation_dict)
# failure_data_df['Air temperature [K]'] = air_imputed_df

# imputation_dict = {
#     'Air temperature [K]': 'median'
#     # 'Process temperature [K]': 'mean',
#     # 'Tool wear [min]': 'median'
# }
# dt = DataTransform(failure_data_df)
# failure_data_df = dt.impute_missing_values(imputation_dict)
# info = DataFrameInfo(failure_data_df)


##############################################################################################################################
# class Plotter:
#     def __init__(self, df):
#         self.df = df
#         # self.data_info = data_info
    
#     def histplot(self, column, kde=False, ax=None):
#         if ax is None:
#             ax = plt.gca()  # Get current active axis if none is provided
#         sns.histplot(self.df[column], kde=kde, ax=ax)
#         ax.set_title(f'Histogram for {column}')

# import seaborn as sns
# import matplotlib.pyplot as plt 
# from statsmodels.graphics.gofplots import qqplot


# class Plotter:
#     def __init__(self, df):
#         self.df = df
    
#     def histplot(self, column, kde=False, ax=None):
#         if ax is None:
#             ax = plt.gca()  # Get current active axis if none is provided
#         sns.histplot(self.df[column], kde=kde, ax=ax)
#         ax.set_title(f'Histogram for {column}')

#     def plot_qq(self, column, ax=None):
#         if ax is None:
#             ax = plt.gca()  # Get current active axis if none is provided
#         qq_plot= sns.qqplot(self.df[column], scale=1, line ='q', ax=ax)
#         plt.title(f'Q-Q plot for {column}')
#         plt.show()

# import matplotlib.pyplot as plt

# plott = Plotter(failure_data_df)
# fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns

# plott.histplot('Air temperature [K]', kde = True, ax = axes[0])
# plott.plot_qq('Air temperature [K]', ax=axes[1])

# # Display the plots
# plt.show()

