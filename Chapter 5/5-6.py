class MLInfrastructure:
    def __init__(self, compute_hours, compute_rate, storage_gb, storage_rate, data_transfer_gb, transfer_rate):
        """
        Initialize the MLInfrastructure with cost components.

        Args:
            compute_hours (float): Number of compute hours required.
            compute_rate (float): Cost per compute hour.
            storage_gb (float): Amount of storage required in GB.
            storage_rate (float): Cost per GB storage.
            data_transfer_gb (float): Data transfer volume in GB.
            transfer_rate (float): Cost per GB data transfer.
        """
        self.compute_hours = compute_hours
        self.compute_rate = compute_rate
        self.storage_gb = storage_gb
        self.storage_rate = storage_rate
        self.data_transfer_gb = data_transfer_gb
        self.transfer_rate = transfer_rate

    def compute_cost(self):
        # Cost for compute resources
        return self.compute_hours * self.compute_rate

    def storage_cost(self):
        # Cost for storage resources
        return self.storage_gb * self.storage_rate

    def data_transfer_cost(self):
        # Cost for data transfer resources
        return self.data_transfer_gb * self.transfer_rate

    def total_cost(self):
        # Total cost is the sum of compute, storage, and data transfer costs.
        return self.compute_cost() + self.storage_cost() + self.data_transfer_cost()


class CloudInfrastructure(MLInfrastructure):
    def __init__(self, compute_hours, storage_gb, data_transfer_gb):
        # Cloud cost rates can be assumed based on current market prices.
        # These rates are in dollars.
        compute_rate = 0.5  # $0.5 per compute hour
        storage_rate = 0.1  # $0.1 per GB storage
        transfer_rate = 0.12  # $0.12 per GB data transfer
        super().__init__(compute_hours, compute_rate, storage_gb, storage_rate, data_transfer_gb, transfer_rate)

    def total_cost(self):
        # Additional cloud overheads can be factored in if needed.
        overhead = 0.05 * super().total_cost()  # 5% overhead
        return super().total_cost() + overhead


class OnPremiseInfrastructure(MLInfrastructure):
    def __init__(self, compute_hours, storage_gb, data_transfer_gb):
        # On-Premise rates might be lower for compute but higher for storage and data management.
        compute_rate = 0.3  # $0.3 per compute hour (lower cost due to capital investment)
        storage_rate = 0.15  # $0.15 per GB storage
        transfer_rate = 0.08  # $0.08 per GB data transfer
        super().__init__(compute_hours, compute_rate, storage_gb, storage_rate, data_transfer_gb, transfer_rate)

    def total_cost(self):
        # For on-premise, consider amortized capital costs as a fixed overhead.
        fixed_overhead = 1000  # Fixed overhead cost for hardware investment
        return super().total_cost() + fixed_overhead


class HybridInfrastructure(MLInfrastructure):
    def __init__(self, compute_hours, storage_gb, data_transfer_gb):
        # Hybrid may combine cloud and on-premise; we use averaged rates.
        compute_rate = (0.5 + 0.3) / 2  # Average compute rate
        storage_rate = (0.1 + 0.15) / 2  # Average storage rate
        transfer_rate = (0.12 + 0.08) / 2  # Average transfer rate
        super().__init__(compute_hours, compute_rate, storage_gb, storage_rate, data_transfer_gb, transfer_rate)

    def total_cost(self):
        # Hybrid may incur integration overheads.
        integration_overhead = 0.03 * super().total_cost()  # 3% overhead
        return super().total_cost() + integration_overhead


# Above code does not have any output. If you want to test that, you can use the following code:
compute_hours = 200  # hours of compute
storage_gb = 500  # storage in GB
data_transfer_gb = 300  # data transfer in GB

# Create infrastructure objects
cloud = CloudInfrastructure(compute_hours, storage_gb, data_transfer_gb)
on_prem = OnPremiseInfrastructure(compute_hours, storage_gb, data_transfer_gb)
hybrid = HybridInfrastructure(compute_hours, storage_gb, data_transfer_gb)

# Print total cost for each
print("Cloud Infrastructure Cost:", cloud.total_cost())
print("On-Premise Infrastructure Cost:", on_prem.total_cost())
print("Hybrid Infrastructure Cost:", hybrid.total_cost())
