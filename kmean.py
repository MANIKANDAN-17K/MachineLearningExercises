import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = pd.read_csv("dataset/Mall_Customers.csv")

X = data[["Annual Income (k$)","Spending Score (1-100)"]]

kmeans = KMeans(n_clusters=5, random_state =42)
y_kmeans = kmeans.fit_predict(X) 

plt.scatter(X.iloc[:,0],X.iloc[:,1],c=y_kmeans)
plt.xlabel("income")
plt.ylabel("Spending Score")
plt.title("K-Means Clustering")
plt.show()

