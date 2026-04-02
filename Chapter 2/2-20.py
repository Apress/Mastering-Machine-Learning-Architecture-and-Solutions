import numpy as np
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

# ---- Minimal sample data ----
# 100 samples, 20 features
X_train = np.random.rand(100, 20)

# Binary target variable
y_train = np.random.randint(0, 2, size=100)

# ---- Feature selection  ----
model = RandomForestClassifier()
rfe = RFE(model, n_features_to_select=10)
X_train_selected = rfe.fit_transform(X_train, y_train)

print("Selected feature shape:", X_train_selected.shape)
