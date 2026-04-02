import pandas as pd
from sklearn.linear_model import LogisticRegression

# Exposed implementation details
data = pd.read_csv("data.csv")
X = data.drop("target", axis=1).fillna(0)
y = data["target"]
model = LogisticRegression(max_iter=100, solver='lbfgs')
model.fit(X, y)
print(model.coef_)
