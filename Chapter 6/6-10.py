class ModelContainer:
    def __init__(self, model, container_name="fraud_detection_container"):
        """
        Initialize the container with the model.

        Args:
            model: The ML model to be containerized.
            container_name (str): A name for the container.
        """
        self.model = model
        self.container_name = container_name
        print(f"Container '{self.container_name}' created for the model.\n")

    def run(self, input_data):
        """
        Simulate running the container to make predictions.

        Args:
            input_data (numpy.ndarray): Input features for prediction.

        Returns:
            numpy.ndarray: Model predictions.
        """
        print(f"Container '{self.container_name}' is running the model...")
        predictions = self.model.predict(input_data)
        print("Container execution complete.\n")
        return predictions


# Wrap the fraud detection model in a container
model_container = ModelContainer(fraud_model)
