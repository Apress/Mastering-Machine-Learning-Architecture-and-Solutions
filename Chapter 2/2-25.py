from sklearn.metrics import confusion_matrix
import numpy as np

# Sample true and predicted labels
y_true = np.array([0, 1, 0, 1, 1, 0, 1])
y_pred = np.array([0, 1, 0, 0, 1, 0, 1])

# Generate confusion matrix
cm = confusion_matrix(y_true, y_pred)
print(cm)
