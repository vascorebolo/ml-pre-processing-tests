# importing libraries
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
import tabulate

def print_line(char = '▀'):
  print(char * 100)

def print_results(method, na_values, df):
  print_line()
  print(f'{method}: ({na_values})')
  print_line()
  print(df.tail(5))

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
print('NA values:')
na_values = df.isna().sum().sum()

# implementation of methods
df_1 = df.bfill(axis = 'columns') # replaces with value of the previous row
print_results('bfill', na_values, df_1)

df_2 = df.ffill(axis = 'columns') # replaces with value of the next row
print_results('ffil', na_values, df_2)

# fillna: Fill NA/NaN values using the specified method.
df_3 = df.fillna(df.mean()) # replaces with value of the mean of the column
print_results('fillna(mean)', na_values, df_3)

df_4 = df.fillna(df.median()) # replaces with value of the median of the column
print_results('fillna(median)', na_values, df_4)

df_5 = x
imputer = KNNImputer()
imputer.fit_transform(df_5)
print_results('KNNImputer', na_values, df_5)

