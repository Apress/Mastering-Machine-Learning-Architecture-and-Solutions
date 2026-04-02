import pandas as pd

# Non-scalable preprocessing
data = pd.read_csv("data.csv")
X = data.drop("target", axis=1)
for col in X.columns:
    X[col] = X[col].fillna(X[col].mean())  # Loops over columns in memory
print(X)

