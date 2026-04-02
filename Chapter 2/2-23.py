import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification

# Sample imbalanced dataset (or use your own X, y)
X, y = make_classification(n_classes=2, class_sep=2,
                           weights=[0.1, 0.9], n_informative=3,
                           n_redundant=1, flip_y=0, n_features=20,
                           n_clusters_per_class=1, n_samples=1000,
                           random_state=42)

# Check class distribution before SMOTE
print(f"Original class distribution: {np.bincount(y)}")

# Initialize SMOTE
smote = SMOTE(random_state=42, sampling_strategy='auto')

# Apply SMOTE
try:
    X_res, y_res = smote.fit_resample(X, y)

    # Check new class distribution
    print(f"Resampled class distribution: {np.bincount(y_res)}")

    # Verify shapes
    print(f"X_res shape: {X_res.shape}, y_res shape: {y_res.shape}")

except Exception as e:
    print(f"Error occurred: {str(e)}")
