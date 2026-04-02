class MonitoringSystem:
    def __init__(self, alert_threshold=0.80):
        """
        Initialize the monitoring system.

        Args:
            alert_threshold (float): Minimum acceptable accuracy.
        """
        self.alert_threshold = alert_threshold
        self.metric_log = []
        print("Monitoring system initialized with alert threshold at {:.2f}%.\n".format(alert_threshold * 100))

    def record_metric(self, accuracy):
        """
        Record the current model accuracy and check if an alert should be triggered.

        Args:
            accuracy (float): The current accuracy of the model.
        """
        self.metric_log.append(accuracy)
        print(f"Recorded model accuracy: {accuracy * 100:.2f}%")
        if accuracy < self.alert_threshold:
            print("ALERT: Model accuracy has dropped below the threshold!\n")

    def display_metrics(self):
        """
        Plot the recorded accuracy metrics over time.
        """
        iterations = range(1, len(self.metric_log) + 1)
        plt.figure(figsize=(8, 4))
        plt.plot(iterations, [acc * 100 for acc in self.metric_log], marker='o', linestyle='-', color='purple')
        plt.xlabel("Iteration")
        plt.ylabel("Accuracy (%)")
        plt.title("Model Accuracy Over Time")
        plt.tight_layout()
        plt.show()
