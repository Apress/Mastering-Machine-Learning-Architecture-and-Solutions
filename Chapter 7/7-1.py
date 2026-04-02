import mlflow
import mlflow.pyfunc

# Load the trained model
model = mlflow.pyfunc.load_model("models:/my_model/Production")

# Register a new version of the model
mlflow.register_model(
    "runs:/<run-id>/my_model",
    "MyModelRegistry"
)
