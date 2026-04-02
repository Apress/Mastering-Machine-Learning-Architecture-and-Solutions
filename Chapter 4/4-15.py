class Preprocessor:
    def __init__(self):
        # Initialize the scaler.
        self.scaler = StandardScaler()
        print("Preprocessor initialized with StandardScaler.")

    def fit_transform(self, X_train):
        """
        Fit the scaler on the training data and transform it.
        Args:
            X_train (np.ndarray): Training features.
        Returns:
            X_scaled (np.ndarray): Scaled training features.
        """
        print("Fitting and transforming training data.")
        X_scaled = self.scaler.fit_transform(X_train)
        return X_scaled

    def transform(self, X_test):
        """
        Transform the test data using the already fitted scaler.
        Args:
            X_test (np.ndarray): Test features.
        Returns:
            X_scaled (np.ndarray): Scaled test features.
        """
        print("Transforming test data.")
        return self.scaler.transform(X_test)

import numpy as np

# Example dummy data
X_train = np.array([[1.0, 2.0],
          [2.0, 3.0],
          [3.0, 4.0]])

X_test = np.array([[1.5, 2.5],
          [2.5, 3.5]])

# Call the Preprocessor class
pre = Preprocessor()

# Fit on training data and transform
X_train_scaled = pre.fit_transform(X_train)
print("Scaled X_train:\n", X_train_scaled)

# Transform test data
X_test_scaled = pre.transform(X_test)
print("Scaled X_test:\n", X_test_scaled)
