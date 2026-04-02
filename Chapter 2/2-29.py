from scipy.stats import ks_2samp
import numpy as np

# Load training and production datasets
train_data = pd.read_csv("train_data.csv")
prod_data = pd.read_csv("prod_data.csv")

# Compare feature distributions
for column in train_data.columns:
    stat, p_value = ks_2samp(train_data[column].dropna(), prod_data[column].dropna())
    if p_value < 0.05:
        print(f"Feature {column} has changed significantly (p={p_value:.5f})")
