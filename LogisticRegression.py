import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix

data = pd.read_csv("dataset/tested.csv")
print(data.head())
print(data.columns)

X = data[["Pclass","Age","Fare"]].fillna(0)
y = data["Survived"]

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)

model = LogisticRegression()
model.fit(x_train,y_train)

pred = model.predict(x_test)

print("Accuracy : ",accuracy_score(y_test,pred))
print("confusion matrix : ",confusion_matrix(y_test,pred))
