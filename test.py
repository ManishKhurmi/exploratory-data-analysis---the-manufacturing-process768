import sys
import os 

sys.path.append(os.path.join(os.getcwd(), '..', 'scripts'))
from scripts import manufacturing_eda_classes

# Import custom Classes 
from scripts.manufacturing_eda_classes import LoadData, DataFrameInfo, DataTransform, Plotter, Models

load_data = LoadData(file_name='failure_data.csv')
df = load_data.df
info = DataFrameInfo(df)
# info.z_score_info()

# print(info.continous_variables())

# print(info.column_names())

# print(info.percentage_of_null())

dt = DataTransform(df)
print(type(dt.unique_observations('Type')))