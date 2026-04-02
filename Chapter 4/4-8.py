import pandas as pd
from sklearn.linear_model import LogisticRegression

# Abstracted training function
def train_model(X, y, params={'max_iter': 100}):
    model = LogisticRegression(**params)
    model.fit(X, y)
    return model

# Usage
data = pd.read_csv("data.csv")
X = data.drop("target", axis=1).fillna(0)
y = data["target"]
model = train_model(X, y)
print(model.coef_)
