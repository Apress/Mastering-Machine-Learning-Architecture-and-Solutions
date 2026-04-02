import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Hardcoded, project-specific preprocessing
data = pd.read_csv("data.csv")
X = data.drop("target", axis=1)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)  # Tied to this dataset
print(X_scaled)
