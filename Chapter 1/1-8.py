import random

# Define a simple search space (you'd have a much larger one in reality)
operations = ["conv3x3", "conv5x5", "maxpool", "avgpool"]
num_layers = 4

def generate_architecture():
    """Generates a random neural network architecture."""
    architecture = []
    for _ in range(num_layers):
        op = random.choice(operations)
        architecture.append(op)
    return architecture

def evaluate_architecture(architecture):
    """Simulates evaluating an architecture (in reality, you'd train it)."""
    # This is a placeholder - replace with actual training and validation
    # For demonstration, assign a random "accuracy" between 0 and 1.
    accuracy = random.uniform(0, 1)
    return accuracy

def simple_nas():
    """Performs a very basic Neural Architecture Search (random search)."""
    num_architectures_to_try = 10  # Try 10 random architectures

    best_architecture = None
    best_accuracy = 0

    for _ in range(num_architectures_to_try):
        architecture = generate_architecture()
        accuracy = evaluate_architecture(architecture)

        print(f"Architecture: {architecture}, Accuracy: {accuracy:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_architecture = architecture

    print(f"\nBest Architecture: {best_architecture}, Best Accuracy: {best_accuracy:.4f}")

# Run the NAS
simple_nas()
