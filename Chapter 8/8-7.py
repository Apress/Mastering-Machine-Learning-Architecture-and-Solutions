import pandas as pd

# Load training and production datasets
train_data = pd.read_csv("train_data.csv")
prod_data = pd.read_csv("production_data.csv")

# Compare feature distributions
for column in train_data.columns:
    train_mean = train_data[column].mean()
    prod_mean = prod_data[column].mean()
    print(f"Feature {column} - Training Mean: {train_mean:.2f}, Production Mean: {prod_mean:.2f}")
