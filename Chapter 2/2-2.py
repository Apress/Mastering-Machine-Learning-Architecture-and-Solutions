# Impute missing values using the median
from sklearn.impute import SimpleImputer
import pandas as pd

# Sample dataset
data = {'age': [25, 30, None, 35, 40], 'salary': [50000, 60000, 55000, None, 70000]}
df = pd.DataFrame(data)

# Create an imputer object
imputer = SimpleImputer(strategy='median')

# Apply imputation
clean_data = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
print(clean_data)
