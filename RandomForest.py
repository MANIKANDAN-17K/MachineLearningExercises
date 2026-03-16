import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data =pd.read_csv("dataset/heart.csv")

x = data.drop("target",axis=1)
y = data["target"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

model = RandomForestClassifier(n_estimators=100)

model.fit(x_train,y_train)

pred = model.predict(x_test)

print("accu : ",accuracy_score(y_test,pred))
