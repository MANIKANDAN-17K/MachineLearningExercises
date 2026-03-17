import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt
data = pd.read_csv("dataset/groceries - groceries.csv")

transactions = []

for i in range(len(data)):
    transactions.append([str(item) for item in data.iloc[i] if str(item) != 'nan'])
te = TransactionEncoder()
te_data = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_data, columns=te.columns_)

freq_items = apriori(df, min_support=0.02,use_colnames=True)

rules = association_rules(freq_items,metric = "confidence",min_threshold=0.3)
print("Frequent Items: \n",freq_items)
print("\nRules : \n",rules)

plt.scatter(rules['support'],rules['confidence'])
plt.xlabel("Support")
plt.ylabel("Confidence")
plt.title("Apriori Rules")
plt.show()
