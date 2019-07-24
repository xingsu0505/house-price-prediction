import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# size of the data 
print("Train data shape:", train.shape)
print("Test data shape:", test.shape)
train.head
plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10,6)

# explore the data and features
train.SalePrice.describe()
# skewness 
print("Skew is:", train.SalePrice.skew())
plt.hist(train.SalePrice, color='blue')
plt.show()
# Apply np.log() to transform train.SalePrice and calculate the skewness a second time
target = np.log(train.SalePrice)
print("Skew is:", target.skew())
plt.hist(target, color='blue')
plt.show()

# Working with numeric features
numeric_features = train.select_dtypes(include=[np.number])
numeric_features.dtypes 

corr = numeric_features.corr()
print(corr['SalePrice'].sort_values(ascending=False)[:5],'\n')
print(corr['SalePrice'].sort_values(ascending=False)[-5:])

# dig deeper on overall quality 
train.OverallQual.unique()
# Build a pivot table of overallqual and saleprice 
quality_pivot = train.pivot_table(index='OverallQual', values='SalePrice', aggfunc=np.median)
quality_pivot
# Make a bar plot of the pivot table 
quality_pivot.plot(kind='bar', color='blue')
plt.xlabel('Overall Quality')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)
plt.show()

# Draw a scatter plot of GrLivArea and SalePrice
plt.scatter(x=train['GrLivArea'], y=target)
plt.ylabel('SalePrice')
plt.xlabel('Above ground living area square feet')
plt.show()

# Do the same for GarageArea
plt.scatter(x=train['GarageArea'], y=target)
plt.xlabel('SalePrice')
plt.ylabel('Garage Area')
plt.show()

# Dealing with outliers 
train = train[train['GarageArea']<1200]
# let's take another look 
plt.scatter(x=train['GarageArea'], y=np.log(train.SalePrice))
plt.xlim(-200,1600)
plt.ylabel('Sale Price')
plt.xlabel('Garage Area')
plt.show()

# Handling null values
nulls = pd.DataFrame(train.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
nulls

# take a look at MiscFeature 
print("Unique values are:", train.MiscFeature.unique())

# Explore non-numeric features 
categoricals = train.select_dtypes(exclude=[np.number])
categoricals.describe()

# one hot encoding for street
print("Original: \n")
print(train.Street.value_counts(), "\n")
train['enc_street'] = pd.get_dummies(train.Street, drop_first=True)
train['enc_street'] = pd.get_dummies(train.Street, drop_first=True)
print('Encoded: \n')
print(train.enc_street.value_counts())

# engineer SaleCondition 
condition_pivot = train.pivot_table(index='SaleCondition', values='SalePrice', aggfunc=np.median)
condition_pivot.plot(kind='bar', color='blue')
plt.xlabel('Sale Condition')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0) # ticks are the values used to show specific points on the coordinates axis
plt.show()

# Partial = 1 , o.w = 0
def encode(x):
    return 1 if x == 'Partial' else 0 
train['enc_condition'] = train.SaleCondition.apply(encode)
test['enc_condition'] = test.SaleCondition.apply(encode)

condition_pivot = train.pivot_table(index='enc_condition', values='SalePrice', aggfunc=np.median)
condition_pivot.plot(kind='bar', color='blue')
plt.xlabel('Encoded Sale Condition')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)
plt.show()

"""
Missing data: before we prepare the data for modeling, we need to deal with missing data.
We'll fill the missing values with an average value and then assign the results to data. 
This is a method of interpolation. We use DataFrame.interpolate() method makes this simple 

"""
data = train.select_dtypes(include=[np.number]).interpolate().dropna()
data.isnull().sum()

# Build a linear model 
# We will assign the features to x and the target variable to y
# We use np.log() above to transform the y variable for the model 
y = np.log(train.SalePrice)
X = data.drop(['SalePrice','Id'], axis=1)

# partition the data and start modeling
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
                        X, y, random_state=42, test_size=0.33
                                                   )

# begin modelling 
from sklearn import linear_model
lr = linear_model.LinearRegression()
# fit the model 
model = lr.fit(X_train, y_train)
# evaluate the performance and visualize results 
# this competition, Kaggle requires us to use root-mean-squared-error 
# we'll also look at the r-squared value 
# r-squared value is a measure of how close the data are to the fitted regression line
# it takes a value between 0 and 1, 1 means that all of the variance in the target is explained 
# by the data, in general, a higher r-squared value means a better fit
# model.score() returns the r-squared value by default 
print("R^2 is: \n", model.score(X_test, y_test)) 
# this return value means that our feature explain approximately 89% of the variance in our target variable

predictions = model.predict(X_test)
from sklearn.metrics import mean_squared_error
print('RMSE is: \n', mean_squared_error(y_test, predictions))

# View the relationship graphically with a scatter plot
actual_values = y_test 
plt.scatter(predictions, actual_values, alpha=0.7, color='b')
plt.xlabel('Predicted Price')
plt.ylabel('Actual Price')
plt.title('Linear Regression Model')
plt.show()

# Improving the model 
# Ridge Regularization 
for i in range(-2,3):
    alpha = 10**i
    rm = linear_model.Ridge(alpha=alpha)
    ridge_model = rm.fit(X_train, y_train)
    preds_ridge = ridge_model.predict(X_test)

    plt.scatter(preds_ridge, actual_values, alpha=0.75, color='b')
    plt.xlabel('Predicted Price')
    plt.ylabel('Actual Price')
    plt.title('Ridge Regularization with alpha = {}'.format(alpha))
    overlay = 'R^2 is: {}\nRMSE is: {}'.format(
                ridge_model.score(X_test,y_test),
                mean_squared_error(y_test,preds_ridge))
    plt.annotate(s=overlay, xy=(12.1,10.6), size='x-large')
    plt.show()

submission = pd.DataFrame()
submission['Id'] = test.Id 
feats = test.select_dtypes(include=[np.number]).drop(['Id'], axis=1).interpolate()
predictions = model.predict(feats)
# transform the predictions to the correct form, reverse log(), we do exp()
final_predictions = np.exp(predictions)
# Look at the difference 
print("Original predictions are: \n", predictions[:5], "\n")
print("Final predictions are: \n", final_predictions[:5])

submission['SalePrice'] = final_predictions
submission.head()

submission.to_csv('submission1.csv', index=False)





















































