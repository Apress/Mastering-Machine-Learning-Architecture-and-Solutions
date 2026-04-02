data_sources = {
    "customer_data": {
        "name": "Customer Database",
        "location": "/path/to/customer_data.csv",
        "format": "CSV",
        "update_frequency": "Daily",
        "description": "Contains customer demographics and purchase history."
    },
    "product_catalog": {
        "name": "Product Catalog API",
        "endpoint": "https://api.example.com/products",
        "format": "JSON",
        "update_frequency": "Real-time",
        "description": "Provides information about available products."
    },
    "sales_data": {
        "name": "Sales Transactions",
        "location": "s3://my-bucket/sales_data/",
        "format": "Parquet",
        "update_frequency": "Hourly",
        "description": "Records of sales transactions."
    }
}

# Accessing information about a specific data source
print(data_sources["customer_data"]["name"])
print(data_sources["product_catalog"]["endpoint"])

# Iterating through all data sources
for source_name, metadata in data_sources.items():
    print(f"\nSource: {source_name}")
    for key, value in metadata.items():
        print(f"  {key}: {value}")

