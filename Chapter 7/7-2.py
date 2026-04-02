from sklearn.linear_model import SGDClassifier
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# 1. Generate an initial small dataset (Simulate original training data)
X_initial, y_initial = make_classification(n_samples=50, n_features=2,
                                           n_informative=2, n_redundant=0,
                                           random_state=42)

# 2. Initialize and Train the Model (The first "full" training)
model = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
# The classes parameter is essential for the first partial_fit call or
# if you are retraining on an existing model that hasn't seen all classes.
model.partial_fit(X_initial, y_initial, classes=np.unique(y_initial))

# 3. Simulate New Incoming Data Batches for Incremental Update
X_new = np.array([[0.5, 0.3], [0.9, 0.8], [0.1, 0.2]])
y_new = np.array([1, 0, 1])

# 4. Perform Incremental Learning (Retraining)
# The model is updated with new data without retraining from scratch.
# We don't need to specify 'classes' here as it was done in the initial fit.
model.partial_fit(X_new, y_new)
