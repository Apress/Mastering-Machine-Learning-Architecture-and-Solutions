def main():
    print("Starting ML pipeline orchestration.")

    # Step 6.1: Load Data
    data_loader = DataLoader()
    X, y = data_loader.load_data()

    # Step 6.2: Split Data into Training and Testing Sets
    print("Splitting data into train and test sets.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 6.3: Preprocess Data
    preprocessor = Preprocessor()
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)

    # Step 6.4: Train the Model
    model_trainer = ModelTrainer()
    model_trainer.train(X_train_scaled, y_train)

    # Step 6.5: Inference - Make Predictions
    predictor = ModelPredictor(model_trainer.get_model())
    predictions = predictor.predict(X_test_scaled)

    # Step 6.6: Evaluate the Model
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
