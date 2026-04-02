import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Everything mixed together
data = pd.read_csv("data.csv")
X = data.drop("target", axis=1).fillna(0)
y = data["target"]
X = (X - X.min()) / (X.max() - X.min())
model = LogisticRegression()
model.fit(X, y)
preds = model.predict(X)
print(f"Accuracy: {accuracy_score(y, preds)}")
