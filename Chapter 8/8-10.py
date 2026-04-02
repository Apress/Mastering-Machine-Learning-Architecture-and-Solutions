class DataLoader:
    def load_data(self):
        """Load and return the Iris dataset."""
        data = load_iris()
        X, y = data.data, data.target
        return X, y


class Preprocessor:
    def __init__(self):
        self.scaler = StandardScaler()

    def fit_transform(self, X_train):
        """Fit the scaler on training data and transform it."""
        return self.scaler.fit_transform(X_train)

    def transform(self, X_test):
        """Transform test data using the fitted scaler."""
        return self.scaler.transform(X_test)


class ModelTrainer:
    def __init__(self):
        self.model = LogisticRegression(max_iter=200)

    def train(self, X_train, y_train):
        """Train the logistic regression model."""
        self.model.fit(X_train, y_train)
        return self.model

    def evaluate(self, X_test, y_test):
        """Evaluate the model using accuracy score."""
        predictions = self.model.predict(X_test)
        return accuracy_score(y_test, predictions)


# Function to orchestrate the ML pipeline
def run_pipeline():
    # Load data
    loader = DataLoader()
    X, y = loader.load_data()

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocess the data
    preprocessor = Preprocessor()
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)

    # Train the model
    trainer = ModelTrainer()
    model = trainer.train(X_train_scaled, y_train)

    # Evaluate the model
    accuracy = trainer.evaluate(X_test_scaled, y_test)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    return model, preprocessor, loader


# Run the pipeline once to ensure it works as expected.
model, preprocessor, loader = run_pipeline()
