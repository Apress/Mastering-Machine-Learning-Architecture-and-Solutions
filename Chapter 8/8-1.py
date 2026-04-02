import pytest
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def test_minmax_scaler():
    scaler = MinMaxScaler()
    X = np.array([[10], [20], [30], [40], [50]])
    X_scaled = scaler.fit_transform(X)

    assert np.all(X_scaled >= 0) and np.all(X_scaled <= 1), "Scaling failed: Values are outside [0,1] range"

test_minmax_scaler()
