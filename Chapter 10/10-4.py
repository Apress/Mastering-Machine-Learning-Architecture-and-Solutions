class EdgeDevice:
    def __init__(self, model):
        self.model = model

    def process_data(self, data):
        # Preprocess data (e.g., sensor readings)
        processed_data = self.preprocess(data)

        # Run inference using the local model
        prediction = self.model.predict(processed_data)  # Replace with actual inference

        # Optionally, send results or updates to a central server
        self.send_to_server(prediction)

        return prediction

    def preprocess(self, data):
        # Implement data preprocessing specific to the edge device
        # For this example, we'll just return the data as is
        return data

    def send_to_server(self, data):
        # Implement logic to send data to a server (e.g., for aggregation)
        print(f"Data sent to server: {data}")

class Model: # A very simple model for the example
    def predict(self, data):
        # Simulate prediction (replace with a real model's prediction)
        return f"Prediction for {data}: {random.random()}"

# Example Usage
my_model = Model() # Replace with your actual model
edge_device = EdgeDevice(my_model)

sensor_data = {"temperature": 25, "humidity": 60}
prediction = edge_device.process_data(sensor_data)
print(f"Edge Device Prediction: {prediction}")

sensor_data_2 = {"temperature": 30, "humidity": 70}
prediction_2 = edge_device.process_data(sensor_data_2)
print(f"Edge Device Prediction 2: {prediction_2}")
