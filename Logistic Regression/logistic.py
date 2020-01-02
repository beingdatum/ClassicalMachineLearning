# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

advert = pd.read_csv(r'advertising.csv')
#Fetching the first few records of the data.
advert.head(10)


advert.info()

advert.describe()

advert['Clicked on Ad'].value_counts()

advert['Male'].value_counts()

advert.isnull().sum()

#EDA
advert.groupby('Clicked on Ad').mean()

advert.groupby(['Clicked on Ad','Male']).size()

sns.set_style('whitegrid')
sns.distplot(advert['Age'], kde = False, bins = 40)

sns.jointplot(x = 'Age', y = 'Area Income', data = advert)
plt.show()

sns.jointplot(x =  'Age', y ='Daily Time Spent on Site', data = advert, kind = 'kde', color = 'red')
plt.show()


sns.countplot(x = 'Male',  data = advert, palette= 'pastel')

sns.pairplot(advert, hue = 'Clicked on Ad')
plt.plot()


#Logistic Regression

X = advert[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage']]
y = advert['Clicked on Ad']


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

from sklearn.linear_model import LogisticRegression
#Creating an instance of Logistic Regression class
logreg = LogisticRegression()
logreg.fit(X_train, y_train)


prediction = logreg.predict(X_test)


from sklearn.metrics import confusion_matrix
conf_Matrix = confusion_matrix(y_test, prediction)
print(conf_Matrix)


from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
print(classification_report(y_test,prediction))
result2 = accuracy_score(y_test,prediction)
print("Accuracy:",result2)








