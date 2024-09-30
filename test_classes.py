from manufacturing_eda_classes import LoadData, DataFrameInfo, DataTransform, Models

print('#'* 100)
load_data = LoadData('failure_data.csv')
failure_data_df = load_data.df
print(failure_data_df.head(2))

print('#'* 100)
info = DataFrameInfo(failure_data_df)
print(info.column_names())

print('#'* 100)
dt = DataTransform(failure_data_df)
type_dummy_df = dt.create_dummies_from_column('Type')
print(type_dummy_df.head(2))

print('#'* 100)
model = Models(failure_data_df)
print(model.chi_squared_test_df(binary_cols=['Machine failure', 'RNF']))

