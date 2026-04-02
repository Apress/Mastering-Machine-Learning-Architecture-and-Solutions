from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from sklearn.datasets import make_classification

# Sample Imbalanced Data
X, y = make_classification(n_samples=100, n_features=2, n_informative=2,
                           n_redundant=0, n_repeated=0, n_classes=2,
                           n_clusters_per_class=1,
                           weights=[0.95, 0.05], # Imbalance: 95% class 0, 5% class 1
                           class_sep=0.8, random_state=0)

# Apply Random Oversampling
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

print("Original class distribution:", Counter(y))
print("Resampled class distribution:", Counter(y_resampled))
