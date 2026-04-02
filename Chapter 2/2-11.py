from sklearn.preprocessing import StandardScaler

# Sample data
X = [[100, 0.1], [200, 0.2], [300, 0.3]]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(X_scaled)
