import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

data = load_iris()

X = data.data

U, S, VT = np.linalg.svd(X)

x_svd = U[:,:2] * S[:2]

plt.scatter(x_svd[:,0],x_svd[:,1])
plt.title("SVD visualization")
plt.show()
