import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

data = pd.read_csv("dataset/Mall_Customers.csv")

X = data[["Annual Income (k$)","Spending Score (1-100)"]]

model = AgglomerativeClustering(n_clusters=5)

labels = model.fit_predict(X)

plt.scatter(X.iloc[:,0],X.iloc[:,1],c=labels)
plt.title("Agglomerative Clustering")
plt.show()
