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

# Correlation matrix
# Heatmaps are great to detect the relations between objects 
corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.8, square=True)
plt.show()

# Zoomed heatmap 
k = 10 # number of variables for heatmap
cols = corrmat.nlargest(k,'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size':10},
                yticklabels=cols.values, xticklabels=cols.values)
plt.show()

# Now let's proceed to scatter plots between SalePrice and correlated variables.
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 
        'FullBath', 'YearBuilt']
sns.pairplot(df_train[cols], size=2.5)
plt.show()

"""
Missing data: imply a reduction of the sample size. This can prevent us from proceeding 
              with the analysis. Moreover, from a substantive perspective, we need to 
              ensure that the missing data process is not biased and hidding the truth 

"""
total = df_train.isnull().sum().sort_values(ascending=True)
percent = (df_train.isnull().sum()/df_train.isna().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

df_train = df_train.drop((missing_data[missing_data['Total']>1]).index,1)
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)
df_train.isnull().sum().max()

"""
Outliers: they can remarkablely affect our model and can be a valuable source of 
info, providing us insights about specific behaviours

"""
# Defind a threshold to detect outliners
# Standardizing data 
saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:, np.newaxis])
low_range = saleprice_scaled[saleprice_scaled[:, 0].argsort()][:10]
high_range = saleprice_scaled[saleprice_scaled[:, 0].argsort()][-10:]
print('outer range (low) of the distribution: ')
print('\nouter range (high) of the distribution: ')
print(high_range)

# Bivariate analysis 
var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
plt.show()

# Deleting points
df_train.sort_values(by = 'GrLivArea', ascending = False)[:2]
df_train = df_train.drop(df_train[df_train[df_train['Id'] == 1299].index)
df_train = df_train.drop(df_train[df_train['Id'] == 524].index)

# Bivariate analysis saleprice and grlivarea
var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', xlim=(0,3000),ylim=(0,800000))
plt.show()

"""

Discover Saleprice in four ways:
1. Normality
2. Homoscedasticity 
3. Linearity
4. Absence of correlated errors

"""

# Histogram and normal probability plot
sns.distplot(df_train['SalePrice'], fit=norm)
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)
plt.show()

# we see that saleprice is not normal distributed, it shows positive skewness, so in case 
# of positive skewness, log transformation usually works well. 
# applying log transformation 
df_train['SalePrice'] = np.log(df_train['SalePrice'])

#transformed histogram and normal probability plot
sns.distplot(df_train['SalePrice'], fit=norm)
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)
plt.show()

# Check with GrLivArea
sns.distplot(df_train['GrLivArea'], fit=norm)
fig = plt.figure()
res = stats.probplot(df_train['GrLivArea'], plot=plt)
# log transform 
df_train['GrLivArea'] = np.log(df_train['GrLivArea'])
# Transformed histogram and normal probability plot 
sns.distplot(df_train['GrLivArea'], fit=norm)
fig = plt.figure()
res = stats.probplot(df_train['GrLivArea'], plot=plt)

# Check with TotalBsmtSF 
sns.distplot(df_train['TotalBsmtSF'], fit=norm)
fig = plt.figure()
res = stats.probplot(df_train['TotalBsmtSF'], plot=plt)

# Observe that in the figure, there are lots of houses don't have basements
# So we decide to create a new binary variable since we cannot perform log 
# tranformation on 0.
df_train['HasBsmt'] = pd.Series(len(df_train['TotalBsmtSF']), index=df_train.index)
df_train['HasBsmt'] = 0 
df_train.loc[df_train['TotalBsmtSF']>0, 'HasBsmt'] = 1
# transform data
df_train.loc[df_train['HasBsmt'] == 1, 'TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])
# histogram and normal probability plot 
sns.distplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm)
fig = plt.figure()
res = stats.probplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)

# homoscedsticity
plt.scatter(df_train['GrLivArea'], df_train['SalePrice'])
plt.scatter(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], df_train['TotalBsmtSF']
            >0['SalePrice'])

# convert categorical variable into indicator variables 
df_train = pd.get_dummies(df_train)
print(df_train)























