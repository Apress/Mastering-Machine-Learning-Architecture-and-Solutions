def main_production_management():
    print("=== Starting Model Management and Updating Simulation ===\n")

    # 1. Initialize the model registry, incremental model, and monitoring system
    registry = ModelRegistry()
    inc_model = IncrementalModel()
    monitor = MonitoringSystem(alert_threshold=0.90)  # Increased threshold for a real dataset

    # 2. Load Real Data and Prepare Batches
    print("Loading Breast Cancer dataset for simulation...")
    data = load_breast_cancer()
    X_full, y_full = data.data, data.target

    # Standardize data (crucial for SGDClassifier performance on real data)
    scaler = StandardScaler()
    X_full = scaler.fit_transform(X_full)

    # Split data into Initial Training (60%) and New Retraining Batch (40%)
    X_initial, X_rest, y_initial, y_rest = train_test_split(X_full, y_full, test_size=0.4, random_state=42)
    # Take 50% of the rest as the "new" batch for retraining (40% * 0.5 = 20% of total)
    X_new, _, y_new, _ = train_test_split(X_rest, y_rest, test_size=0.5, random_state=24)

    print(f"Initial Training Samples: {len(X_initial)}")
    print(f"Incremental Update Samples: {len(X_new)}\n")

    # 3. Perform initial training and register version 1.0.0
    initial_acc = inc_model.initial_train(X_initial, y_initial)
    version1 = ModelVersion(version="1.0.0",
                            model=inc_model.model,
                            accuracy=initial_acc,
                            metadata={"dataset": "Breast_Cancer", "samples": len(X_initial)})
    registry.register(version1)
    monitor.record_metric(initial_acc)

    # Simulate waiting period before new data arrives
    print("Waiting for new data for retraining...\n")
    time.sleep(1)

    # 4. Perform incremental update (retraining) and register version 1.1.0
    # Note: SGDClassifier performs poorly if the data changes drastically,
    # so we expect a small performance change, which is realistic.
    new_acc = inc_model.incremental_update(X_new, y_new)
    version2 = ModelVersion(version="1.1.0",
                            model=inc_model.model,
                            accuracy=new_acc,
                            metadata={"dataset": "Breast_Cancer", "new_samples": len(X_new)})
    registry.register(version2)
    monitor.record_metric(new_acc)

    # 5. List all registered model versions
    registry.list_versions()

    # 6. Display monitoring metrics over iterations
    monitor.display_metrics()

    print("=== Simulation Complete ===")


# Execute the simulation if this cell is run directly
if __name__ == "__main__":
    main_production_management()
