if best_model_name == "Random Forest":
    param_grid = {
        'model__n_estimators': [50, 100, 200],
        'model__max_depth': [None, 10, 20],
        'model__min_samples_split': [2, 5, 10]
    }
elif best_model_name == "Logistic Regression":
    param_grid = {
        'model__C': [0.01, 0.1, 1, 10],
        'model__solver': ['liblinear', 'lbfgs']
    }
else:  # Support Vector Machine or other models
    param_grid = {
        'model__C': [0.1, 1, 10],
        'model__kernel': ['linear', 'rbf']
    }

# GridSearchCV
grid_search = GridSearchCV(best_pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train_smote, y_train_smote)

# Bayesian Optimization (BayesSearchCV)
bayes_search = BayesSearchCV(best_pipeline, param_grid, n_iter=10, cv=5, scoring='roc_auc', n_jobs=-1)
bayes_search.fit(X_train_smote, y_train_smote)

print(f"Best Hyperparameters (GridSearch): {grid_search.best_params_}")
print(f"Best Hyperparameters (Bayesian): {bayes_search.best_params_}")

# Choose the best estimator from both methods
optimized_model = grid_search.best_estimator_ if grid_search.best_score_ > bayes_search.best_score_ else bayes_search.best_estimator_
