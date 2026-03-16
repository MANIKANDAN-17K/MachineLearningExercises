import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = pd.read_csv("dataset/Student_Performance.csv")

print(data.head())
data["Extracurricular Activities"] = data["Extracurricular Activities"].map({"Yes" : 1,"No":0})
X = data[["Hours Studied"]]
y = data["Performance Index"]

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = LinearRegression()
model.fit(x_train,y_train)

pred = model.predict(x_test)

print("MSE : ",mean_squared_error(y_test,pred))

plt.scatter(X,y,color = "blue")
plt.plot(X,model.predict(X),color="red")
plt.xlabel("Hours Studied")
plt.ylabel("Exam score")
plt.title("Linear Regression")
plt.show()

