import pandas as pd
from sklearn.linear_model import LogisticRegression

# One function does everything and accesses data directly
def process_and_train():
    data = pd.read_csv("data.csv")  # Direct data access
    X = data.drop("target", axis=1)
    X["feature1"] = X["feature1"].fillna(0)  # Specific preprocessing
    y = data["target"]
    model = LogisticRegression()
    model.fit(X, y)
    return model

model = process_and_train()
