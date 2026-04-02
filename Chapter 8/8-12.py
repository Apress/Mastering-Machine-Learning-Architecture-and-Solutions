def simulate_data_pipeline_error():
    print("Simulating data pipeline error: feature dimension mismatch.")
    # Load data correctly first
    X, y = loader.load_data()

    # Introduce error: remove one column from X_train to simulate a dimension mismatch
    X_err = X[:, :-1]  # Remove the last feature column
    try:
        # Attempt to split and preprocess the faulty data
        X_train_err, X_test_err, y_train_err, y_test_err = train_test_split(X_err, y, test_size=0.2, random_state=42)
        X_train_scaled_err = preprocessor.fit_transform(X_train_err)
        X_test_scaled_err = preprocessor.transform(X_test_err)
        # Train the model with erroneous data
        trainer = ModelTrainer()
        trainer.train(X_train_scaled_err, y_train_err)
    except Exception as e:
        logging.error("Error during pipeline execution: %s", e)
        print("Debug Tip: Ensure that the input feature dimensions match across all pipeline components.")


# Run the error simulation
simulate_data_pipeline_error()
