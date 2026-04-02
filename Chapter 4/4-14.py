from sklearn.datasets import load_iris
import numpy as np


class DataLoader:

    def __init__(self):
        print("DataLoader initialized.")

    def load_data(self):
        """
        Load and return the Iris dataset.

        Returns:
          X (np.ndarray): Features.
          y (np.ndarray): Labels.
        """
        print("Loading Iris dataset.")
        data = load_iris()
        X, y = data.data, data.target
        print(f"Dataset loaded with {X.shape[0]} samples and {X.shape[1]} features.")
        return X, y


# ---- Call the function ----
loader = DataLoader()
X, y = loader.load_data()
