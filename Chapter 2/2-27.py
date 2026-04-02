import pandas as pd

# Load dataset
df = pd.read_csv("data.csv")

# Check for missing values
missing_values = df.isnull().sum()
print("Missing Values:\n", missing_values[missing_values > 0])

# Check for unexpected data types
print("\nData Types:\n", df.dtypes)
