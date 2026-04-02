import matplotlib.pyplot as plt

plt.hist(train_data["predictions"], alpha=0.5, label="Training")
plt.hist(prod_data["predictions"], alpha=0.5, label="Production")
plt.legend()
plt.xlabel("Prediction Scores")
plt.ylabel("Frequency")
plt.title("Concept Drift Detection")
plt.show()
