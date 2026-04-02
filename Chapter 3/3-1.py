# Grid search example
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.datasets import load_iris

# Load sample dataset
data = load_iris()
X, y = data.data, data.target

# Define models and parameters
models = {
    'RandomForest': RandomForestClassifier(),
    'SVM': SVC()
}

params = {
    'RandomForest': {'n_estimators': [10, 50, 100]},
    'SVM': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
}

# Perform Grid Search
for model_name in models:
    grid_search = GridSearchCV(models[model_name], params[model_name], cv=5)
    grid_search.fit(X, y)
    print(f"Best parameters for {model_name}: {grid_search.best_params_}")
