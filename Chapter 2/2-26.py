from pyspark.sql import SparkSession

# Create a Spark session
spark = SparkSession.builder.appName("DataPipelineScaling").getOrCreate()

# Load a CSV file into a DataFrame
df = spark.read.csv("large_dataset.csv", header=True, inferSchema=True)

# Perform transformations
df_filtered = df.filter(df['value'] > 100)

# Show the results
df_filtered.show()
