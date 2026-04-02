def main_testing_debugging():
    print("=== Starting Testing and Debugging Simulation ===\n")

    # Run unit tests to ensure the pipeline components are working correctly.
    print("Running unit tests...\n")
    run_unit_tests()

    # Simulate a data pipeline error and demonstrate debugging.
    print("\nSimulating a data pipeline error to demonstrate debugging...\n")
    simulate_data_pipeline_error()

    print("\n=== Testing and Debugging Simulation Complete ===")


# Execute the orchestration if this cell is run directly.
if __name__ == "__main__":
    main_testing_debugging()
