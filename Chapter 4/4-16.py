class ModelTrainer:
    def __init__(self):
        # Using a simple logistic regression for demonstration.
        self.model = LogisticRegression(max_iter=200)
        print("ModelTrainer initialized with LogisticRegression.")

    def train(self, X_train, y_train):
        """
        Train the model on the training data.
        Args:
            X_train (np.ndarray): Processed training features.
            y_train (np.ndarray): Training labels.
        """
        print("Training model.")
        self.model.fit(X_train, y_train)
        print("Model training complete.")

    def get_model(self):
        """
        Return the trained model.
        """
        return self.model

# Dummy example data
X_train = np.array([[0.1, 0.5],
          [0.2, 0.4],
          [0.3, 0.3]])
y_train = np.array([0, 1, 0])

trainer = ModelTrainer()
trainer.train(X_train, y_train)
model = trainer.get_model()

print("Trained Model:", model)
