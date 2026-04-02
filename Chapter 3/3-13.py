best_model_name = max(cv_results, key=cv_results.get)
best_pipeline = pipelines[best_model_name]
print(f"\nBest Model Selected: {best_model_name}")
print("\nPerforming hyperparameter optimization for the best model...")
