
#Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importing the mall dataset
dataset = pd.read_csv("mall.csv")
X = dataset.iloc[:,[3,4]].values

#Using the dendrogram to find optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram= sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title("Dendrogram")
plt.xlabel("Customers")
plt.ylabel("Euclidean Distance")

plt.show()

#Fitting hierarchical clustering to the mall dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = "euclidean", linkage="ward")
y_hc = hc.fit_predict(X)

#Visualizing the clusters
plt.scatter(X[y_hc==0, 0], X[y_hc ==0,1], s = 100, c="red", label = "Cluster 1")
plt.scatter(X[y_hc==1, 0], X[y_hc ==1,1], s = 100, c="blue", label = "Cluster 2")
plt.scatter(X[y_hc==2, 0], X[y_hc ==2,1], s = 100, c="green", label = "Cluster 3")
plt.scatter(X[y_hc==3, 0], X[y_hc ==3,1], s = 100, c="yellow", label = "Cluster 4")
plt.scatter(X[y_hc==4, 0], X[y_hc ==4,1], s = 100, c="cyan", label = "Cluster 5")

plt.title("Cluster of Clients")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score(0-100)")
plt.legend()
plt.show()