class HardwareSimulator:
    def __init__(self):
        print("Hardware Simulator initialized.")

    def simulate_performance(self, hardware_type, workload_size):
        """
        Simulate the performance (e.g., training time) of a given hardware type.

        Args:
            hardware_type (str): 'GPU' or 'TPU'.
            workload_size (float): A factor representing the workload size (e.g., data size or model complexity).

        Returns:
            float: Simulated training time (in hours).
        """
        base_time = workload_size  # Base time proportional to workload size
        if hardware_type.upper() == 'GPU':
            # Assume GPU takes base_time multiplied by a factor.
            performance_factor = 1.2  # GPU is slightly slower than TPU for this simulation
        elif hardware_type.upper() == 'TPU':
            performance_factor = 0.8  # TPU is faster
        else:
            raise ValueError("Invalid hardware type. Choose 'GPU' or 'TPU'.")

        simulated_time = base_time * performance_factor
        print(f"Simulated training time using {hardware_type.upper()}: {simulated_time:.2f} hours.")
        return simulated_time


# To generate some output we can add the following code:
sim = HardwareSimulator()
sim.simulate_performance("GPU", 10)
sim.simulate_performance("TPU", 10)
