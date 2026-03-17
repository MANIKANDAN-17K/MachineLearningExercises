import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from sklearn.datasets import load_iris

data = load_iris()
X = data.data
y = data.target

ica = FastICA(n_components=2, random_state=42)

x_ica = ica.fit_transform(X)

plt.scatter(x_ica[:,0],x_ica[:,1],c=y)
plt.title("ICA Visualization")
plt.show()
