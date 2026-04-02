import dask.dataframe as dd
import pandas as pd

# Convert pandas DataFrame to Dask DataFrame for distributed processing
dask_data = dd.from_pandas(data_cleaned, npartitions=2)

# Perform operations on Dask DataFrame, filling numeric missing values
dask_data = dask_data.map_partitions(
  lambda df: df.fillna(df.median(numeric_only=True)),
  meta=data_cleaned
)

print("\nPipeline is ready for scaling with distributed frameworks or cloud platforms!")
