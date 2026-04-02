import pandas as pd
from sklearn.linear_model import LogisticRegression
# Cohesive preprocessing module

def preprocess(X):
    return X.fillna(0)

# Cohesive training module
def train_model(X, y):
    model = LogisticRegression()
    model.fit(X, y)
    return model

# Usage with loose coupling
data = pd.read_csv("data.csv")
X_raw = data.drop("target", axis=1)
y = data["target"]
X_processed = preprocess(X_raw)
