#Federated Learning
import random

class Server:
    def __init__(self):
        self.global_model = None  # Initialize later

    def aggregate_updates(self, client_updates):
        # Simple averaging for demonstration (replace with more sophisticated aggregation)
        avg_weights = {}
        for key in client_updates[0].keys():            avg_weights[key] = sum(update[key] for update in client_updates) / len(client_updates)
        self.global_model = avg_weights  # Update the global model

class Client:
    def __init__(self, data):
        self.data = data
        self.local_model = {} # Initialize a model (a simple dictionary for now)

    def train_local_model(self):
        # Simulate local training (replace with actual training)
        for feature in self.data:
            self.local_model[feature] = random.random() # Assign random weights for demonstration

    def send_update(self, server):
        return self.local_model  # Send the updated model weights

# Example usage
server = Server()
server.global_model = {"feature1": 0.5, "feature2": 0.2, "feature3": 0.8} # Initialize the global model

# Sample Data (replace with your actual data)
data1 = ["feature1", "feature2"]
data2 = ["feature2", "feature3"]
data3 = ["feature1", "feature3"]

clients = [Client(data1), Client(data2), Client(data3)]

for _ in range(3):  # Simulate a few rounds of training
    client_updates = []
    for client in clients:
        client.local_model = server.global_model.copy() # Start training with the global model
        client.train_local_model()
        update = client.send_update(server)
        client_updates.append(update)

    server.aggregate_updates(client_updates)
    print(f"Global Model (Round {_+1}): {server.global_model}")
