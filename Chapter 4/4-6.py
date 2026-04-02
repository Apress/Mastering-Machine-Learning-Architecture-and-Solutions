import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Reusable preprocessing function
def scale_features(X, scaler=None):
    if scaler is None:
        scaler = MinMaxScaler()
        scaler.fit(X)
    return scaler.transform(X), scaler

# Usage
data = pd.read_csv("data.csv")
X = data.drop("target", axis=1)
X_scaled, scaler = scale_features(X)
print(X_scaled)
# Reusable scaler for new data
new_data = pd.read_csv("new_data.csv")
new_scaled = scale_features(new_data, scaler)[0]
