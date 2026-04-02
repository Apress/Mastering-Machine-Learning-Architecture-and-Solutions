from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification  # For sample data

# Generate some sample data (replace with your actual data)
X, y = make_classification(n_samples=100, n_features=20, random_state=42)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model and hyperparameters (CORRECTED)
rand_params = {
    'n_estimators': [10, 50, 100, 200],  # Values are now provided!
    'max_depth': [None, 10, 20, 30],
    'max_features': ['sqrt', 'log2', None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestClassifier(random_state=42)
random_search = RandomizedSearchCV(rf, rand_params, n_iter=10, cv=5, n_jobs=-1, verbose=1, random_state=42)

random_search.fit(X_train, y_train)

print(f"Best parameters: {random_search.best_params_}")

# Evaluate the best model on the test set
best_rf = random_search.best_estimator_
test_accuracy = best_rf.score(X_test, y_test)
print(f"Test Accuracy: {test_accuracy}")
