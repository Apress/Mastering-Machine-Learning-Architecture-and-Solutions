print("\nCreating pipelines for preprocessing and modeling...")
pipelines = {}
for model_name, model in models.items():
    pipelines[model_name] = Pipeline([
        ('scaler', StandardScaler()),  # Feature scaling
        ('model', model)
    ])

