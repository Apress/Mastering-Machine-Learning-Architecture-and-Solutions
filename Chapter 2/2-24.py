import numpy as np
from sklearn.linear_model import LogisticRegression

# ---- Minimal sample data----
# 200 samples, 5 features
X = np.random.rand(200, 5)

# Imbalanced binary target: mostly 0s, few 1s
y = np.random.choice([0, 1], size=200, p=[0.9, 0.1])

# ---- Cost-sensitive Logistic Regression ----
model = LogisticRegression(class_weight='balanced')
model.fit(X, y)

print("Model coefficients:", model.coef_)
