import matplotlib.pyplot as plt

# Simulated loss values from model training
epochs = range(1, 21)
train_loss = [0.4, 0.35, 0.30, 0.25, 0.22, 0.20, 0.18, 0.15, 0.14, 0.13, 0.12, 0.11, 0.10, 0.095, 0.09, 0.085, 0.08, 0.075, 0.07, 0.065]
val_loss = [0.42, 0.38, 0.36, 0.34, 0.35, 0.37, 0.39, 0.40, 0.42, 0.45, 0.47, 0.50, 0.52, 0.55, 0.57, 0.60, 0.62, 0.65, 0.67, 0.70]

plt.plot(epochs, train_loss, 'b-', label="Training Loss")
plt.plot(epochs, val_loss, 'r-', label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Overfitting Detection")
plt.legend()
plt.show()
