from manufacturing_eda_classes import LoadData, DataFrameInfo, DataTransform

if __name__=='__main__':

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