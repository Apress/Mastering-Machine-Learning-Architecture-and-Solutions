print("\nPerforming cross-validation for model selection...")
cv_results = {}
for model_name, pipeline in tqdm(pipelines.items(), desc="Cross-validating models", unit="model"):
    if model_name == "Logistic Regression":
        pipeline.set_params(model__class_weight=class_weights)  # Apply class weighting

    # Perform cross-validation with progress bar for each fold
    scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='roc_auc')
    cv_results[model_name] = scores.mean()
    print(f"\n{model_name} - Mean ROC AUC: {scores.mean():.4f}")

