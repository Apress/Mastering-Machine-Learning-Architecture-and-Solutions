class MonitoringSystem:
    def __init__(self):
        # Initialize an empty list to store metric logs.
        self.metrics_log = []
        print("Monitoring system initialized.\n")

    def record_metrics(self, latency, error_rate):
        """
        Record performance metrics.

        Args:
            latency (float): Simulated response time of the deployed model.
            error_rate (float): Simulated error rate of model predictions.
        """
        metric = {"latency": latency, "error_rate": error_rate}
        self.metrics_log.append(metric)
        print(f"Recorded metrics: Latency = {latency} sec, Error Rate = {error_rate}%")

    def display_metrics(self):
        # Plot the recorded metrics for visualization.
        if not self.metrics_log:
            print("No metrics recorded.")
            return

        latencies = [m["latency"] for m in self.metrics_log]
        error_rates = [m["error_rate"] for m in self.metrics_log]
        iterations = range(1, len(self.metrics_log) + 1)

        # Plot latency
        plt.figure(figsize=(8, 4))
        plt.plot(iterations, latencies, marker='o', linestyle='-', color='blue')
        plt.xlabel("Iteration")
        plt.ylabel("Latency (sec)")
        plt.title("Model Response Latency Over Time")
        plt.tight_layout()
        plt.show()  # Figure 4: Latency Monitoring Plot

        # Plot error rate
        plt.figure(figsize=(8, 4))
        plt.plot(iterations, error_rates, marker='o', linestyle='-', color='red')
        plt.xlabel("Iteration")
        plt.ylabel("Error Rate (%)")
        plt.title("Model Prediction Error Rate Over Time")
        plt.tight_layout()
        plt.show()  # Figure 5: Error Rate Monitoring Plot


# Instantiate the monitoring system
monitoring_system = MonitoringSystem()

# Simulate periodic monitoring by recording metrics
# For the purpose of demonstration, we record metrics over 5 iterations.
for i in range(5):
    simulated_latency = np.random.uniform(0.2, 1.0)  # simulated latency between 0.2 and 1.0 seconds
    simulated_error_rate = np.random.uniform(0, 5)  # simulated error rate between 0% and 5%
    monitoring_system.record_metrics(simulated_latency, simulated_error_rate)
    time.sleep(0.5)  # Simulate time gap between recordings

