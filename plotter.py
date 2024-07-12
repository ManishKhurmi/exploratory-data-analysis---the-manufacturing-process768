# MISSING Data Analayis
import pandas as pd 
import DataFrameInfo_class as info
from scipy.stats import normaltest 
from statsmodels.graphics.gofplots import qqplot 

import matplotlib.pyplot as plt

failure_data = pd.read_csv('failure_data_after_data_transformation.csv')

##########################################################################################
# failure_data.head()
# failure_data.info()


information = info.DataFrameInfo(failure_data)


failure_data.columns
# Filter for columns with null values 
null_variable_bool = information.percentage_of_null() > 0
percentage_of_nulls_per_column = information.percentage_of_null()[null_variable_bool]
percentage_of_nulls_per_column
columns_with_null_names = percentage_of_nulls_per_column.index.tolist()
columns_with_null_names

# MCAR / MAR / NMAR ?
# Are the missing values mostly along a certain 'Type' of product? 
    # boolean on null values
    # filter data for data only with NULL 
    # calculate the %NULL per Type by doing a count since they are all binary columns 
    # repeat for each variable 

failure_data.columns

#idea
#failure_data['Air temperature [K]'].isnull()

#func
def null_mask(df, column_name):
    return df[column_name].isnull()    

air_temp_null_bool = null_mask(failure_data, 'Air temperature [K]')
air_temp_null_bool

filtered_air_temp_null = failure_data[air_temp_null_bool]
filtered_air_temp_null[['H', 'L', 'M']].sum() / len(filtered_air_temp_null) * 100 

# 60 % of the NULLS in air_temp are in the 'L' Type, 28% in 'M'
##########################################################################################
def null_mask(df, column_name):
    return df[column_name].isnull()   

def percentage_of_nulls_per_quality_type(df, column_name):
    air_temp_null_bool = null_mask(df, column_name)
    filtered_air_temp_null = failure_data[air_temp_null_bool]
    return filtered_air_temp_null[['H', 'L', 'M']].sum() / len(filtered_air_temp_null) * 100 

# test 
for i in columns_with_null_names:
    print(i ,':')
    print(percentage_of_nulls_per_quality_type(failure_data, i))
    print('////////')

# The null values for each variable of interest are concentrated arount the 'Low' quality type, this does not look like MCAR 

#  This is a nice to do, however not worth the work:
    # check if there is a bias in the overall data set for the number of Quality Types 
        # If there is then this means that deleting the data might cause a bias in our data set, e.g. the NULL values center around the 'L' product type & the Product types are balanceed
    # consider imputing the data on mean of each product type, this only holds if there is a considerable variance of values across the Types for each NULL column of interest 
    # Shortcut would be to impute based on the mean & drop the column with 4% values 

statistics_across_null_columns = information.describe_statistics()[columns_with_null_names]
# What percentage of the data is Type?

failure_data['Type'].value_counts() / len(failure_data) * 100
# This split almost exactly matches the split in the NULL values data we found 

# Go ahead with further analysis 
##########################################################################################
# test the normalist of each column of interest 

# plot a histogram for 'Air temperature'
failure_data['Air temperature [K]'].hist(bins = 40)
plt.show()


# Print a histogram for all of the columns of interest 
# columns_with_null_names

# for i in columns_with_null_names:
#     failure_data[i].hist(bins=40)
#     plt.show()

# D'Agostino's K^2 Test for 'Air temperature' 
    # null hypothesis = the distribution is not normally distributed 
    # a p-value with a confidence level of less than 0.05 is significant evidence that the data are normally distributed 

stat, p = normaltest(failure_data['Air temperature [K]'], nan_policy = 'omit')
print('Statistics=%.3f, p=%.3f' % (stat, p))
# data is strongly normally distributed. 

# Use the Q-Q plot to see if 'Air temperature' follows a certain theoretical distribution 

# scale 'Air Temperature'

qq_plot_air_temp = qqplot(failure_data['Air temperature [K]'], scale=1, line ='q')
plt.show()
# Analysis: 
    # From the Q-Q plot it is clear that the data are normally distributed through the middle of the range and the upper quartiles. 
    # There are deviations from the lower bounds of the data. As this is the average air temperature of the room, the likelyhood is that as more products are manufactured they each add a small incriment in the average air temperature rising. 
    # The data is normally distributed from the middle and upper bounds of the There are deviations from the lower bound. 

# Mean & Median of 'Air Temperature'

print(f'The median of Air Temperature [K] is {failure_data["Air temperature [K]"].median()}')
print(f'The median of Air Temperature [K] is {failure_data["Air temperature [K]"].mean()}')

# There is not much difference, so we will go with the median of the data 
# TODO: impute using mean 
##############################################################################################################
# Repeat this for 'Process temperature [K]', 'Tool wear [min]'
# fucntion 

class Plotter: 
    def __init__(self, df):
        self.df = df 
    
    def plot_hist(self, column_name):
        self.df[column_name].hist(bins = 40)
        plt.show()

    def normal_test(self, column_name):
        stat, p = normaltest(self.df[column_name], nan_policy = 'omit')
        print('Statistics=%.3f, p=%.3f' % (stat, p))
    
    def plot_qq(self, column_name):
        qq_plot_air_temp = qqplot(self.df[column_name], scale=1, line ='q')
        plt.show()

    def print_mean(self, column_name):
        print(f'The mean of {column_name} is {self.df[column_name].mean()}')
    
    def print_median(self, column_name):
        print(f'The median of {column_name} is {self.df[column_name].median()}')

# Testing 
# plotter = Plotter(failure_data)
# plotter.plot_qq('Air temperature [K]')
# plotter.print_mean('Air temperature [K]')
# plotter.print_mode('Air temperature [K]')

# Test normal_test
# plotter = Plotter(failure_data)
# plotter.normal_test('Air temperature [K]')

# Test plot_hist 
# plotter = Plotter(failure_data)
# plotter.plot_hist('Air temperature [K]')
columns_with_null_names

#######################################################
# repeat analysis for 'Process temperature [K]', 'Tool wear [min]'
# Instance  
plotter = Plotter(failure_data)

process_temp_hist = plotter.plot_hist('Process temperature [K]')
process_temp_hist
# Data looks normally distributed 

process_temp_normal_test = plotter.normal_test('Process temperature [K]')
process_temp_normal_test
# p-value is 0, data is normally distributed 
# check qq plot 

process_temp_qq_plot = plotter.plot_qq('Process temperature [K]')
process_temp_qq_plot
# Most of the lower quartile, middle range and upper quartile fit the data. 
# There is slight deviation at the beginning of the lower quartile and at the end of the upper quartile. This is not significant enough to warrant further tests. 

# impute using mean or median

process_temp_mean = plotter.print_mean('Process temperature [K]')
process_temp_median = plotter.print_median('Process temperature [K]')

# TODO: impute using mean 

#######################################################
# repeat analysis for 'Tool wear [min]'

# Instance
plotter = Plotter(failure_data)

tool_wear_hist = plotter.plot_hist('Tool wear [min]')
tool_wear_hist
# Data is not normally distributed 
# try changing the number of bins used in the plot 

tool_wear_hist_more_bins = failure_data['Tool wear [min]'].hist(bins = 100)
plt.show()
# There are consistent spikers throughout the hole data set, this may be due to the 'High" quality tool type lasting longer 

tool_wear_normal_test = plotter.normal_test('Tool wear [min]')
# p-value is showing 0, however we know visually that this is not the case 

tool_wear_qq_plot = plotter.plot_qq('Tool wear [min]')
tool_wear_qq_plot
# significant deviations in the lower quartile. 
# The middle and upper quartile seem to show tendency towards being normally distributed, however, this is rather mis leading after viewing the histogram

tool_wear_mean = plotter.print_mean('Tool wear [min]')
tool_wear_median = plotter.print_median('Tool wear [min]')
# the mean and median are close. We will use the median 

tool_wear_range = failure_data['Tool wear [min]'].max() - failure_data['Tool wear [min]'].min()
tool_wear_range
failure_data['Tool wear [min]'].describe()

##################################################################################################################################
# Data Transformation decision
    # Given the that the range is quite large. I would consider imputing based on the average of 'Tool wear [min]' split by Types. 
        # However in the interest of time,  given that this variable only has 4% of missing data and the data set is statistically large, we can drop these rows 

# Data Transformations based on MISSING Data Analayis

# Treating the missing data
# For `Air temperature [K]` and `Process temperature [K]` impute using the median 

failure_data['Air temperature [K]'] = failure_data['Air temperature [K]'].fillna(failure_data['Air temperature [K]'].median())
failure_data['Process temperature [K]'] = failure_data['Process temperature [K]'].fillna(failure_data['Process temperature [K]'].median())

# check that the NULL treatment worked 
percentage_of_nulls_per_column


# For `Tool wear [min]` NULL values, drop rows 
# tool_wear_null_bool = failure_data['Tool wear [min]'].isnull()

# failure_data = failure_data[tool_wear_null_bool]

# percentage_of_nulls_per_column


# failure_data['Tool wear [min]'].dropna() 

failure_data.dropna(subset='Tool wear [min]', inplace=True)

failure_data


# worked 












    



