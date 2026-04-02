from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# Define the model and hyperparameters
grid_params = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf']
}

svc = SVC()
grid_search = GridSearchCV(svc, grid_params, cv=5)

grid_search.fit(X, y)
print(f"Best parameters: {grid_search.best_params_}")
