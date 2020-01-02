# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
#Importing Libraries
import pandas as pd
from sklearn.ensemble  import RandomForestClassifier
from sklearn.model_selection import train_test_split

#Reading data
data = pd.read_csv(r"Social_Network_Ads.csv")
data.head()

#Removing User ID, as it's an increamental value that doesn't add to our classification prediction
data.pop('User ID')

#Replacing categorical values to numericals
data['Gender'].replace(['Male','Female'],[1,0],inplace=True)

#Using features: Gender, Age for prediction of Purchased label
feature_cols = ['Gender', 'Age']
X = data[feature_cols] # Features
y = data.Purchased # Target variable

#Divide the data into train and test split. 
#The following code will split the dataset into 70% training data and 30% of testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

#Train the model with the help of DecisionTreeClassifie
clf = RandomForestClassifier(n_estimators = 500)
clf = clf.fit(X_train,y_train)

#At last we need to make prediction. It can be done with the help of following script −
y_pred = clf.predict(X_test)


#Next, we can get the accuracy score, confusion matrix and classification report as follows −
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_test,y_pred)
print("Accuracy:",result2)
