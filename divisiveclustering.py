import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
data = pd.read_csv("dataset/Mall_Customers.csv")
X = data[["Annual Income (k$)","Spending Score (1-100)"]]

kmeans = KMeans(n_clusters = 2)
labels = kmeans.fit_predict(X)

plt.scatter(X.iloc[:,0],X.iloc[:,1], c= labels)

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=200,c='red',marker='X')
plt.xlabel("annual income")
plt.ylabel("Spending score")
plt.title("divisive Clustering ")
plt.show()
