# Simulated learning curves for underfitting
train_acc = [0.5, 0.52, 0.54, 0.56, 0.58, 0.60, 0.61, 0.62, 0.63, 0.64]
val_acc = [0.48, 0.49, 0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 0.55, 0.56]

plt.plot(range(1, 11), train_acc, 'b-', label="Training Accuracy")
plt.plot(range(1, 11), val_acc, 'r-', label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Underfitting Detection")
plt.legend()
plt.show()
