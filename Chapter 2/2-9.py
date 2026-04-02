from sklearn.preprocessing import PolynomialFeatures
import numpy as np

# Sample data
X = np.array([[2, 3], [3, 4], [5, 6]])

# Create polynomial features (degree 2)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
print(X_poly)
