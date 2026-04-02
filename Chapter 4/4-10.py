from pyspark.sql import SparkSession
from pyspark.sql.functions import col, mean

# Scalable preprocessing with PySpark
spark = SparkSession.builder.appName("ScalableML").getOrCreate()
data = spark.read.csv("data.csv", header=True, inferSchema=True)
X = data.drop("target")
means = {col: X.select(mean(col).alias("mean")).collect()[0]["mean"] for col in X.columns}
X = X.na.fill(means)
X.show()
spark.stop()
