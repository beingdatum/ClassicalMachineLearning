# -*- coding: utf-8 -*-

#Importing Libraries
import pandas as pd
import numpy as np
from sklearn import svm, datasets
import matplotlib.pyplot as plt

#Load the input data
iris = datasets.load_iris()

#Taking first two features as variables
X = iris.data[:, :2]
y = iris.target

#Plot the SVM boundaries with original data
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h = (x_max / x_min)/100
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
X_plot = np.c_[xx.ravel(), yy.ravel()]

#Value of regularization parameter
C = 1.0

#SVM classifier object
Svc_classifier = svm.SVC(kernel='linear', C=C).fit(X, y)
Z = Svc_classifier.predict(X_plot)
Z = Z.reshape(xx.shape)
plt.figure(figsize=(15, 5))
plt.subplot(121)
plt.contourf(xx, yy, Z, cmap=plt.cm.tab10, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.title('Support Vector Classifier with linear kernel')