import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


data = pd.read_csv("dataset/iris.csv")

print(data.head())
print(data.columns)

X = data.drop("species",axis=1)
y = data["species"]

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

model = DecisionTreeClassifier()
model.fit(x_train,y_train)

pred = model.predict(x_test)

print("Accuracy : ",accuracy_score(y_test,pred))

plt.figure(figsize=(10,6))
plot_tree(model,filled = True)
plt.show()
