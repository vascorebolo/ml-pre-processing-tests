# importing libraries
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
import tabulate

def print_line(char = '▀'):
  print(char * 100)

# loading data
df = pd.read_csv('BostonHousing.csv')
y = df['medv'] # get medv column: Median value of owner-occupied homes in $1000's
# drop
# Remove rows or columns by specifying label names and corresponding axis
# index: single label or list-like
# inplace: bool, default False - If False, return a copy. Otherwise, do operation in place and return None.
df.drop('medv', axis = 1, inplace = True)

x = df
print_line()
print(x.shape, y.shape)


# Return a boolean same-sized object indicating if the values are NA.
# NA values, such as None or numpy.NaN

print_line('_')
print(df.isna().sum().sum())

# implementation of methods
df_1 = df.bfill(axis = 'columns') # replaces with value of the previous row
print_line()
print('bfill')
print_line()
print(df_1)

df_2 = df.ffill(axis = 'columns') # replaces with value of the next row
print_line()
print('ffill')
print_line()
print(df_2)

# fillna: Fill NA/NaN values using the specified method.
# df_3 = df.fillna(df.mean()) # replaces with value of the mean of the column
# print_line()
# print(df_3)

# df_4 = df.fillna(df.median()) # replaces with value of the median of the column
# print_line()
# print(df_4)
