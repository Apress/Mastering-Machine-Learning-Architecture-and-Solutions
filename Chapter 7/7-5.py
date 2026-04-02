class IncrementalModel:
    def __init__(self):
        # Initialize the SGDClassifier with partial_fit support
        self.model = SGDClassifier(max_iter=1000, tol=1e-3)
        self.classes_ = np.array([0, 1])  # Binary classification
        print("Initialized Incremental Model.\n")

    def initial_train(self, X, y):
        """
        Train the model initially using a training dataset.

        Args:
            X (numpy.ndarray): Training features.
            y (numpy.ndarray): Training labels.
        """
        print("Performing initial training of the model...")
        self.model.partial_fit(X, y, classes=self.classes_)
        acc = accuracy_score(y, self.model.predict(X))
        print(f"Initial training complete. Accuracy: {acc * 100:.2f}%\n")
        return acc

    def incremental_update(self, X_new, y_new):
        """
        Update the model with new training data using incremental learning.

        Args:
            X_new (numpy.ndarray): New training features.
            y_new (numpy.ndarray): New training labels.
        """
        print("Performing incremental update on the model...")
        self.model.partial_fit(X_new, y_new)
        acc = accuracy_score(y_new, self.model.predict(X_new))
        print(f"Incremental update complete. New data accuracy: {acc * 100:.2f}%\n")
        return acc
