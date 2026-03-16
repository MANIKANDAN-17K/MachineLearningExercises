import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

data = pd.read_excel("dataset/Groceries_dataset.xlsx")
print(data.head())

encoder = LabelEncoder()

data["itemDescription"] = encoder.fit_transform(data["itemDescription"])

print(data.head())

X = data[["Member_number"]]
Y = data[["itemDescription"]]

X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size = 0.2)
model = LogisticRegression()
model.fit(X_train,y_train)
print("model trained successfully")
print(data.columns)

y_pred = model.predict(X_test);

print("accuracy : ",accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))