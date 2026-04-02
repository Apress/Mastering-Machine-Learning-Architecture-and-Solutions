class DeploymentPipeline:
    def __init__(self, container):
        """
        Initialize the deployment pipeline with the containerized model.

        Args:
            container (ModelContainer): The container holding the model.
        """
        self.container = container

    def build(self):
        print("Building the container image...")
        time.sleep(1)  # Simulate build time delay.
        print("Container image build complete.\n")

    def test(self):
        print("Running tests on the container...")
        time.sleep(1)  # Simulate testing time delay.
        # For demonstration, assume tests always pass.
        print("All container tests passed.\n")

    def deploy(self):
        print("Deploying the container to production...")
        time.sleep(1)  # Simulate deployment time delay.
        print("Container deployed successfully.\n")

    def run_pipeline(self, input_data):
        """
        Run the entire deployment pipeline and execute the container.

        Args:
            input_data (numpy.ndarray): Data for the model to process.

        Returns:
            numpy.ndarray: Predictions from the deployed container.
        """
        self.build()
        self.test()
        self.deploy()
        # Run the container to get predictions.
        predictions = self.container.run(input_data)
        return predictions


# Create a deployment pipeline instance using the containerized model
pipeline = DeploymentPipeline(model_container)

# Simulate some input data for prediction (e.g., 10 samples with dummy features)
input_data = np.random.rand(10, 4)
