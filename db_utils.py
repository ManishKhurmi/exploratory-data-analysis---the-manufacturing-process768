# Code to extract data from database 

# Step 1
    # class RDSDatabaseConnector:

# imports
import yaml

# Parcing a YAML file with Python
with open('credentials.yaml') as file:
        credentials = yaml.safe_load(file)

# Check the Keys are what we expect 
for i in credentials:
        print(i)

# Step 3: After installing the package create a function which loads the credentials.yaml file and returns the data dictionary contained within. 
# This will be be passed to your RDSDatabaseConnector as an argument which the class will use to connect to the remote database.

# create a function that parses yaml files into python and returns a dictionary 
def yaml_to_dict(yaml_file):
        with open(yaml_file) as file:
               return yaml.safe_load(file)

# Test with the "credentials.yaml" file               
credentials_dict = yaml_to_dict(yaml_file='credentials.yaml')
credentials_dict        
        
######################################################################
# Task 2 Extract the data from the RDS database 
    # Step 4 - create __init__
    # Step 5 - initialise SQLAlchemy 

# Getting the data from the Data Base 

# Testing the connection before adding it as a method in the class
from sqlalchemy import create_engine
import pandas as pd 

RDS_HOST = 'eda-projects.cq2e8zno855e.eu-west-1.rds.amazonaws.com'
RDS_PASSWORD = 'EDAprocessanalysis'
RDS_USER = 'manufacturinganalyst'
RDS_DATABASE = 'process_data'
RDS_PORT = '5432'
DATABASE_TYPE = 'postgresql'
DBAPI = 'psycopg2'

engine = create_engine(f"{DATABASE_TYPE}+{DBAPI}://{RDS_USER}:{RDS_PASSWORD}@{RDS_HOST}:{RDS_PORT}/{RDS_DATABASE}")
engine.execution_options(isolation_level='AUTOCOMMIT').connect()

# find out the table names in the data base 
from sqlalchemy import inspect
inspector = inspect(engine)
inspector.get_table_names() # the table name we want is failure_data


# Step 6 - extract data from RDS and return it as a pandas dataframe 
failure_data = pd.read_sql_table('failure_data', engine)

# Check the head of the failure_data set 
failure_data.head()

# Step 7 - save the data in a csv format 
# Export data frame to CSV 
failure_data.to_csv('failure_data.csv')

########################################################################################################################
# Come back to this after successfully downloading the data

credentials_dict['RDS_HOST']
engine.execution_options(isolation_level='AUTOCOMMIT').connect()


class RDSDatabaseConnector(credentials_dict):
        '''
        This Class is used to connect to the AWS RDS Database
        '''
        def __init__(self, credentials_dict):
                self.credentials_dict = credentials_dict

        
########################################################################################################################
# Task 3 

# load the cvs data 
failure_data = pd.read_csv('failure_data')

# Shape of the data frame 
failure_data.shape

# Print the head of the data frame 
failure_data.head()


                 


