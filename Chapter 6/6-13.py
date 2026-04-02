def main_deployment():
    print("=== Starting the Automated Deployment and Monitoring Simulation ===\n")

    # Run the deployment pipeline to deploy the containerized model and obtain predictions.
    predictions = pipeline.run_pipeline(input_data)
    print("Predictions from the deployed model:")
    print(predictions, "\n")

    # After deployment, simulate monitoring of the deployed model.
    print("Simulating monitoring of the deployed model...\n")
    # Display the recorded monitoring metrics.
    monitoring_system.display_metrics()

    print("=== Simulation Complete ===")


# Execute the main deployment simulation
if __name__ == "__main__":
    main_deployment()
