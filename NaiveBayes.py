import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer

data = pd.read_csv("dataset/spam.csv",encoding = "latin-1")

data = data.iloc[:, :2]
data.columns = ["Category", "Message"]

data.dropna(inplace=True)

X = data["Message"]
y = data["Category"]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state = 42)

model = MultinomialNB()
model.fit(x_train,y_train)

pred = model.predict(x_test)
print("accuracy : ",accuracy_score(y_test,pred))
