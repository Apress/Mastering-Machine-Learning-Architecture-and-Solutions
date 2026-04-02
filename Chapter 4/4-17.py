class ModelPredictor:
    def __init__(self, model):
        """
        Initialize with a trained model.
        Args:
            model: A trained machine learning model.
        """
        self.model = model
        print("ModelPredictor initialized.")

    def predict(self, X_test):
        """
        Generate predictions for the test data.
        Args:
            X_test (np.ndarray): Processed test features.
        Returns:
            predictions (np.ndarray): Predicted labels.
        """
        print("Making predictions.")
        predictions = self.model.predict(X_test)
        return predictions

predictor = ModelPredictor(model)
preds = predictor.predict(X_test)
print("Predictions:", preds)
