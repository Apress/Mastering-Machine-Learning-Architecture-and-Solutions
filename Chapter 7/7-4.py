class ModelVersion:
    def __init__(self, version, model, accuracy, metadata=None):
        """
        Initialize a model version.

        Args:
            version (str): Version number (e.g., "1.0.0").
            model: The trained model object.
            accuracy (float): Model accuracy on a validation set.
            metadata (dict, optional): Additional metadata (e.g., training time).
        """
        self.version = version
        self.model = model
        self.accuracy = accuracy
        self.metadata = metadata or {}
        self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    def __str__(self):
        return f"Version: {self.version}, Accuracy: {self.accuracy * 100:.2f}%, Timestamp: {self.timestamp}, Metadata: {self.metadata}"


class ModelRegistry:
    def __init__(self):
        self.registry = []
        print("Model Registry initialized.\n")

    def register(self, model_version):
        """
        Register a new model version.

        Args:
            model_version (ModelVersion): The model version to register.
        """
        self.registry.append(model_version)
        print(f"Registered model version: {model_version}\n")

    def list_versions(self):
        """
        List all registered model versions.
        """
        print("Listing all model versions:")
        for version in self.registry:
            print(version)
