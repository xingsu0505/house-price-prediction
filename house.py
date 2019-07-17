import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
from scipy.stats import norm 
from sklearn.preprocessing import StandardScaler 
from scipy import stats 
import warnings 
warnings.filterwarnings('ignore')

# Import the file 
df_train = pd.read_csv('..//house-price//train.csv')
df_train.shape
df_train.columns
df_train.head()

"""
What expectations do people have in mind when buying a house?
LotArea 
Utilities 
OverallQual 
OverallCond 
YearBuilt 
Neighborhood

"""
# Let's do some research on SalePrice 
df_train['SalePrice'].describe() # returns a dataframe of useful info 
# Draw a histogram 
sns.distplot(df_train['SalePrice'])
plt.show()

# The graph shows positive skewness and looks like a irregular normal distribution 
print('Skewness: %f' % df_train['SalePrice'].skew())
print('Kurtosis: %f' %df_train['SalePrice'].kurt()) # kurtosis means the sharpness of the peak

# Explore relevant variables 
var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'],df_train[var]], axis = 1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))

# Scatter plot of totalbsmtsf/saleprice
var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
# Totalbsmtsf indicates a quadratic relationship with saleprice 

# now let's look at the relationship between overallqual and saleprice
var = 'OverallQual'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(8,6))
figure = sns.boxplot(x=var, y='SalePrice', data=data)
figure.axis(ymin=0, ymax=800000)

# look at year built 
var = 'YearBuilt'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplot(figsize=(16,8))
fig = sns.boxplot(x=var, y='SalePrice', data=data)
fig.axis(ymin=0, ymax=800000)
plt.xticks(rotation=90)
