import boto3

# Define the AWS region explicitly
# NOTE: Replace 'us-east-1' with the region you want to query.
REGION = 'us-east-1'

# Initialize EC2 client with the specified region
ec2 = boto3.client('ec2', region_name=REGION)

# Retrieve available GPU instance types
try:
    gpu_instances = ec2.describe_instance_types(
        Filters=[{'Name': 'processor-info.supported-gpus', 'Values': ['NVIDIA']}]
    )

    # Print instance details
    for instance in gpu_instances['InstanceTypes']:
        instance_type = instance.get('InstanceType', 'N/A')
        vcpus = instance.get('VCpuInfo', {}).get('DefaultVCpus', 'N/A')
        memory_mib = instance.get('MemoryInfo', {}).get('SizeInMiB', 'N/A')

        print(f"Instance Type: {instance_type}, vCPUs: {vcpus}, Memory: {memory_mib} MiB")

except Exception as e:
    # Handle potential exceptions like invalid region or authorization issues
    print(f"An error occurred: {e}")
