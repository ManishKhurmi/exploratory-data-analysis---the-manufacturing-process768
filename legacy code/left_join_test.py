import numpy as np
import pandas as pd
np.random.seed(0)
# transactions
left_df = pd.DataFrame({'transaction_id': ['A', 'B', 'C', 'D'], 
                       'user_id': ['Peter', 'John', 'John', 'Anna'],
                       'value': np.random.randn(4),
                      })
# users
right_df = pd.DataFrame({'user_id': ['Paul', 'Mary', 'John',
                                     'Anna'],
                        'favorite_color': ['blue', 'blue', 'red', 
                                           np.NaN],
                       })

print(left_df)
print('')
print(' ')
print(right_df)




joined_df = pd.merge(left_df, right_df, left_index=True, right_index=True)
joined_df


import numpy as np
import pandas as pd
np.random.seed(0)

# Create a function 
def left_join_dataframes(left_df, right_df, left_index = True, right_index = True):
    joined_df = pd.merge(left_df, right_df, left_index=True, right_index=True)
    return joined_df

df_a = left_df
df_b = right_df



left_join_dataframes(left_df=df_a, right_df=df_b)


# More testing 
import numpy as np
import pandas as pd
np.random.seed(0)

# Create a function 
def left_join_dataframes(left_df, right_df, left_index = True, right_index = True):
    joined_df = pd.merge(left_df, right_df, left_index=True, right_index=True)
    return joined_df

# TODO: replace the function in DT_v2.py

df1 = pd.DataFrame({'a':range(6),
                    'b':[5,3,6,9,2,4]}, index=list('abcdef'))
print(df1)

df2 = pd.DataFrame({'c':range(4),
                    'd':[10,20,30, 40]}, index=list('abhi'))

print(df2)

left_join_dataframes(left_df=df1, right_df=df2) # works as expected


# Imports
import pandas as pd

# Load the Data into a data frame 
failure_data = pd.read_csv("failure_data.csv")
#failure_data.head()

# Create a class for Data Transformations 

class DataTransform:
    def __init__(self, df):
        self.df = df

    def left_join_dataframes(self.df, right_df, left_index=True, right_index=True):
        '''
        This functions joins on the index of the LEFT DataFrame
        '''
        joined_df = pd.merge(self.df, right_df, left_index=True, right_index=True)
        return joined_df
    

