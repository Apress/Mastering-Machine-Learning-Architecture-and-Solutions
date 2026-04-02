#K-Means Clustering Example
from sklearn.cluster import KMeans
import numpy as np

# Sample data
X = np.array([[1, 1], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])

# Create and train the K-Means model (e.g., 2 clusters)
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(X)

# Get cluster labels for each data point
labels = kmeans.labels_

# Print cluster assignments
print(f"Cluster Labels: {labels}")

# Get the coordinates of the cluster centers
centroids = kmeans.cluster_centers_
print(f"Centroids: {centroids}")

# Predict the cluster for new data points
new_data = np.array([[2, 2], [7, 9]])
predictions = kmeans.predict(new_data)
print(f"Predictions for new data: {predictions}")

Expected Results
Cluster Labels: [1 1 0 0 1 0]
Centroids: [[7.33333333 9.        ]
 [1.16666667 1.13333333]]
Predictions for new data: [1 0]
