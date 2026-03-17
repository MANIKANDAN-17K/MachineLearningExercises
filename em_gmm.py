import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

data = pd.read_csv("dataset/Mall_Customers.csv")

X = data[["Annual Income (k$)","Spending Score (1-100)"]]

gmm = GaussianMixture(n_components = 5)
labels = gmm.fit_predict(X)

plt.scatter(X.iloc[:,0],X.iloc[:,1], c=labels)
plt.title("EM (GMM)")
plt.show()
