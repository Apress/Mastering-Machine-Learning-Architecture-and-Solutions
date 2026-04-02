iris = load_iris()
X = iris.data
y = iris.target

# For simplicity, we will convert it into a binary classification problem
# (e.g., class 0 vs. classes 1 and 2).
y_binary = (y == 0).astype(int)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

print("Data loaded and split:")
print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")
