def main_simulation():
    print("Starting ML Infrastructure and Hardware Performance Simulation.\n")

    # Define workload parameters
    compute_hours = 500
    storage_gb = 200
    data_transfer_gb = 150
    workload_size = 10

    print("Workload Parameters:")
    print(f"Compute Hours: {compute_hours}")
    print(f"Storage (GB): {storage_gb}")
    print(f"Data Transfer (GB): {data_transfer_gb}")
    print(f"Workload Size Factor: {workload_size}\n")

    # Instantiate and compute cost for Cloud Infrastructure
    cloud = CloudInfrastructure(compute_hours, storage_gb, data_transfer_gb)
    cloud_cost = cloud.total_cost()
    print(f"Estimated Cloud Infrastructure Cost: ${cloud_cost:.2f}")

    on_prem = OnPremiseInfrastructure(compute_hours, storage_gb, data_transfer_gb)
    on_prem_cost = on_prem.total_cost()
    print(f"Estimated On-Premise Infrastructure Cost: ${on_prem_cost:.2f}")

    hybrid = HybridInfrastructure(compute_hours, storage_gb, data_transfer_gb)
    hybrid_cost = hybrid.total_cost()
    print(f"Estimated Hybrid Infrastructure Cost: ${hybrid_cost:.2f}\n")

    hardware_sim = HardwareSimulator()
    gpu_time = hardware_sim.simulate_performance("GPU", workload_size)
    tpu_time = hardware_sim.simulate_performance("TPU", workload_size)

    infrastructures = ['Cloud', 'On-Premise', 'Hybrid']
    costs = [cloud_cost, on_prem_cost, hybrid_cost]

    plt.figure(figsize=(8, 5))
    plt.bar(infrastructures, costs, color='skyblue')
    plt.xlabel('Infrastructure Type')
    plt.ylabel('Estimated Cost ($)')
    plt.title('Cost Comparison of ML Infrastructures')
    plt.tight_layout()
    plt.show()  # Figure 3: Cost Comparison Bar Chart

    hardware = ['GPU', 'TPU']
    times = [gpu_time, tpu_time]

    plt.figure(figsize=(8, 5))
    plt.bar(hardware, times, color='lightgreen')
    plt.xlabel('Hardware Type')
    plt.ylabel('Simulated Training Time (hours)')
    plt.title('Hardware Performance Comparison')
    plt.tight_layout()
    plt.show()  # Figure 4: Hardware Performance Comparison Bar Chart


if __name__ == "__main__":
    main_simulation()
