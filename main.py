# importing libraries
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

# loading data
df = pd.read_csv('BostonHousing.csv')
y = df['medv'] # get medv column: Median value of owner-occupied homes in $1000's
print(y.shape)
# df.drop('medv', axis = 1, inplace = True)
# X = df
# print(X.shape, y.shape)
# print(y)
# print(df.isna().sum().sum())