# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('carprice.csv')

#Removing few features to make the solution less complicated
#The below features can be converted into OneHotEncoder and the results can be enhanced
dataset.pop('fueltype')
dataset.pop('aspiration')
dataset.pop('drivewheel')
dataset.pop('enginelocation')
dataset.pop('enginetype')
dataset.pop('fuelsystem')
dataset.pop('CarName')

#Let's simply convert these categorical to numerical's using LabelEncoder/replace functionality
dataset['doornumber'].replace(['four','two'],[4,2],inplace=True)
dataset['carbody'].replace(['sedan','hatchback', 'wagon', 'hardtop', 'convertible'],[1,2,3,4,5],inplace=True)
dataset['cylindernumber'].replace(['four','six','five', 'eight', 'two', 'three', 'twelve'],[4,6,5,8,2,3,12],inplace=True)

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 18].values

#No use of OneHotEncoder here, as we have removed the categorical fields.
'''
# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 2] = labelencoder.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [2])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]
'''

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

#Let's plot Actual vs Predicted
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.scatter(y_test, y_pred)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
plt.show()



#Building optimal model using Backward Elimination
import statsmodels.api as sm

X = np.append(arr = np.ones((205, 1)).astype(int), values = X, axis = 1)
#Step 2 of backward elimination
X_opt = X[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17, 18]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,1,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17, 18]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,1,2,5,6,7,8,9,10,11,12,13,14,15,16,17, 18]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,1,2,8,9,11,12,13,14,15,16,17, 18]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,1,2,8,9,11,12,14,15,16,17]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,1,2,8,11,12,14,15,16,17]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
