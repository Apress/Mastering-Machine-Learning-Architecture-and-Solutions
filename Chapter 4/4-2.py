import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Preprocessing module
def preprocess_data(data):
    X = data.drop("target", axis=1).fillna(0)
    return (X - X.min()) / (X.max() - X.min()), data["target"]

# Training module
def train_model(X, y):
    model = LogisticRegression()
    model.fit(X, y)
    return model

# Evaluation module
def evaluate_model(model, X, y):
    preds = model.predict(X)
    return accuracy_score(y, preds)

# Usage
data = pd.read_csv("data.csv")
X, y = preprocess_data(data)
model = train_model(X, y)
accuracy = evaluate_model(model, X, y)
print(f"Accuracy: {accuracy}")
