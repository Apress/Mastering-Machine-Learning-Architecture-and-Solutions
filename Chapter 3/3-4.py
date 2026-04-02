import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification

# Sample Data (replace with your actual data)
X, y = make_classification(n_samples=100, n_features=10, random_state=42)

def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 10, 200)
    max_depth = trial.suggest_int('max_depth', 3, 10) #Or suggest_categorical(['None', 3, 10]) if you want None as an option
    # ... other hyperparameters ...

    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    score = cross_val_score(rf, X, y, cv=5, scoring='accuracy').mean() # Use cross-validation

    return score

study = optuna.create_study(direction='maximize')  # Maximize accuracy
study.optimize(objective, n_trials=10) # Number of trials

print(f"Best parameters: {study.best_params}")
print(f"Best score: {study.best_value}")
