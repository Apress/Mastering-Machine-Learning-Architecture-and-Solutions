import numpy as np
from sklearn.preprocessing import StandardScaler

# Sample feature matrix X
X = np.array([
    [1.0, 10.0, 100.0],
    [2.0, 20.0, 200.0],
    [3.0, 30.0, 300.0],
    [4.0, 40.0, 400.0]
])

# Manually define train/test split for the example
X_train = X[:3]   # first 3 rows = training features
X_test = X[3:]    # last row = test features

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("X_train_scaled:\n", X_train_scaled)
print("X_test_scaled:\n", X_test_scaled)
Expected result:
X_train_scaled:
 [[-1.22474487 -1.22474487 -1.22474487]
 [ 0.          0.          0.        ]
 [ 1.22474487  1.22474487  1.22474487]]
X_test_scaled:
 [[2.44948974 2.44948974 2.44948974]]
