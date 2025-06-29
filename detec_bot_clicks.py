# read data from the ongage_activity table in SQL Server
df = spark.read \
    .format("jdbc") \
    .option("url", "jdbc:sqlserver://XXXX;databaseName=DataOps_Data") \
    .option("dbtable", "XXX") \
    .option("user", "XXX") \
    .option("password", "XXXX") \
    .option("encrypt", "true") \
    .option("trustServerCertificate", "true") \
    .load()

# write the data to the ongage_activity table in Databricks
df.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("pocn_data.silver.ongage_activity")

# Filter for 'click' activities
df_clicks = df.filter(df.activity_type == 'click')
display(df_clicks.limit(5))

from pyspark.sql.functions import monotonically_increasing_id
# Add an index column
df_clicks = df_clicks.withColumn("index_col", monotonically_increasing_id())

from pyspark.sql.functions import col, regexp_extract

# Define the regular expression pattern for the timestamp
pattern = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{4}"

# Filter the DataFrame to find rows that don't match the pattern
# Using regexp_extract, we can extract the matching part of the string and then compare it to the original
# If the extracted part is empty, it means the original string didn't match the pattern
non_matching_df = df_clicks.withColumn("extracted", regexp_extract(col("campaign_timestamp"), pattern, 0)) \
                           .filter(col("extracted") == "")

# Show the non-matching rows
non_matching_df.select("campaign_timestamp").show(truncate=False)

from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from pyspark.sql.functions import to_timestamp
import re

# UDF to normalize the timestamp
@udf(StringType())
def normalize_timestamp(ts):
    # This regex will match the fractional part of the timestamp
    match = re.search(r'(\.\d+)', ts)
    if match:
        # Ensures the fractional part of the seconds is four digits by adding trailing zeros if necessary
        normalized_fraction = match.group(1).ljust(5, '0')
        # Replace the original fractional part with the normalized one
        return re.sub(r'(\.\d+)', normalized_fraction, ts)
    else:
        # If there is no fractional part, add ".0000" to match the format
        return ts + ".0000"

# Apply the UDF to normalize the timestamps
df_clicks = df_clicks.withColumn(
    "normalized_campaign_timestamp",
    normalize_timestamp(col("campaign_timestamp"))
)

# Now convert the normalized timestamp to an actual timestamp type
df_clicks = df_clicks.withColumn(
    "campaign_timestamp_normalized",
    to_timestamp(col("normalized_campaign_timestamp"), "yyyy-MM-dd HH:mm:ss.SSSS")
)

from pyspark.sql.functions import year, month, dayofmonth, dayofweek, hour

# Add time-based features
df_clicks = df_clicks.withColumn("year", year("campaign_timestamp_normalized"))
df_clicks = df_clicks.withColumn("month", month("campaign_timestamp_normalized"))
df_clicks = df_clicks.withColumn("day_of_month", dayofmonth("campaign_timestamp_normalized"))
df_clicks = df_clicks.withColumn("day_of_week", dayofweek("campaign_timestamp_normalized"))  # Monday=1, Sunday=7
df_clicks = df_clicks.withColumn("hour_of_day", hour("campaign_timestamp_normalized"))

from pyspark.sql.window import Window
from pyspark.sql.functions import collect_set, size
# Quantify user interaction with the campaign, such as the number of clicks per user and frequency of activity.
windowSpec = Window.partitionBy("email")

df_clicks = df_clicks.withColumn("user_total_clicks", size(collect_set("campaign_data").over(windowSpec)))
df_clicks = df_clicks.withColumn("user_total_campaigns", size(collect_set("campaign_name").over(windowSpec)))

# Identify the number of unique users or activities associated with each IP address.
ipWindowSpec = Window.partitionBy("ip_address")
df_clicks = df_clicks.withColumn("ip_total_clicks", size(collect_set("campaign_data").over(ipWindowSpec)))
df_clicks = df_clicks.withColumn("ip_total_users", size(collect_set("email").over(ipWindowSpec)))

from pyspark.sql.functions import lag, unix_timestamp, avg, second
from pyspark.sql.types import TimestampType
from pyspark.sql.functions import when

# Convert the timestamp string to TimestampType with microsecond precision
df_clicks = df_clicks.withColumn("campaign_timestamp_normalized", col("campaign_timestamp_normalized").cast(TimestampType()))

# Use the window specification to define the ordering
windowSpec = Window.partitionBy("email").orderBy("campaign_timestamp_normalized")

# Add a column for the previous timestamp
df_clicks = df_clicks.withColumn("prev_timestamp", lag("campaign_timestamp_normalized").over(windowSpec))

# Calculate the time difference in microseconds
df_clicks = df_clicks.withColumn(
    "time_diff_microseconds",
    (col("campaign_timestamp_normalized").cast("long") * 1000000 + second("campaign_timestamp_normalized") * 1000 - 
     (col("prev_timestamp").cast("long") * 1000000 + second("prev_timestamp") * 1000))
)

# Calculate the time difference in seconds
df_clicks = df_clicks.withColumn("time_diff_seconds", unix_timestamp("campaign_timestamp_normalized") - unix_timestamp("prev_timestamp"))

# Calculate the average time difference, excluding the nulls
windowSpecWithoutOrder = Window.partitionBy("email")
df_clicks = df_clicks.withColumn("avg_time_between_clicks", avg("time_diff_seconds").over(windowSpecWithoutOrder))
df_clicks = df_clicks.withColumn("avg_time_between_clicks_microseconds", avg("time_diff_microseconds").over(windowSpecWithoutOrder))

# Fill the initial null value of 'time_diff_seconds' with the user's/IP's average time difference
df_clicks = df_clicks.withColumn(
    "time_diff_seconds",
    when(col("time_diff_seconds").isNull(), col("avg_time_between_clicks")).otherwise(col("time_diff_seconds"))
)

# Fill the initial null value of 'time_diff_microseconds' with the user's/IP's average time difference
df_clicks = df_clicks.withColumn(
    "time_diff_microseconds",
    when(col("time_diff_microseconds").isNull(), col("avg_time_between_clicks_microseconds")).otherwise(col("time_diff_microseconds"))
)

# Track how many times the same link is clicked by a user or IP.
from pyspark.sql.functions import count
linkWindowSpec = Window.partitionBy("link_id")
df_clicks = df_clicks.withColumn("link_total_clicks", count("link_id").over(linkWindowSpec))

# Assess the diversity of browsers and campaigns per user or IP.
df_clicks = df_clicks.withColumn("user_browser_diversity", size(collect_set("browser").over(windowSpec)))
df_clicks = df_clicks.withColumn("ip_campaign_diversity", size(collect_set("campaign_name").over(ipWindowSpec)))

numeric_features = [
    "hour_of_day", "user_total_clicks", "time_diff_seconds", "time_diff_microseconds", "avg_time_between_clicks", "avg_time_between_clicks_microseconds" 
]
features_df = df_clicks.select(*numeric_features,"index_col")
# Check for remaining null values before using VectorAssembler
for column in numeric_features:
    null_count = features_df.filter(col(column).isNull()).count()
    if null_count > 0:
        print(f"Column {column} still has {null_count} null values")


from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import Imputer

# List of input features
numeric_features = [
    "time_diff_seconds", "time_diff_microseconds", "avg_time_between_clicks", "avg_time_between_clicks_microseconds"
]

features_df = df_clicks.select(*numeric_features,"index_col")

# Initialize the Imputer
imputer = Imputer(
    inputCols=["time_diff_seconds", "avg_time_between_clicks", "time_diff_microseconds", "avg_time_between_clicks_microseconds"], 
    outputCols=["time_diff_seconds_imputed", "avg_time_between_clicks_imputed", "time_diff_microseconds_imputed", "avg_time_between_clicks_microseconds_imputed"]
).setStrategy("mean")

# Apply the Imputer to the DataFrame
features_df = imputer.fit(features_df).transform(features_df)

# To keep the same column names
features_df = features_df.drop("time_diff_seconds", "avg_time_between_clicks", "time_diff_microseconds", "avg_time_between_clicks_microseconds") \
                         .withColumnRenamed("time_diff_seconds_imputed", "time_diff_seconds") \
                         .withColumnRenamed("avg_time_between_clicks_imputed", "avg_time_between_clicks") \
                         .withColumnRenamed("time_diff_microseconds_imputed", "time_diff_microseconds") \
                         .withColumnRenamed("avg_time_between_clicks_microseconds_imputed", "avg_time_between_clicks_microseconds")     

# Initialize the VectorAssembler
assembler = VectorAssembler(inputCols=numeric_features, outputCol="features")

# Transform the features into a single vector column
features_vector_df = assembler.transform(features_df)

# Now you have a new column 'features' which is a vector of all your input features
# This DataFrame can be used as input to fit an ML model
#features_vector_df.show()

features_vector_df = features_vector_df.withColumn("index_col", df_clicks["index_col"])

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Convert the 'features' column in the Spark DataFrame to a Pandas DataFrame
pandas_features_df = features_vector_df.select('features','index_col').toPandas()

# Convert the 'features' vector column to an array suitable for Scikit-learn
features_array = np.array(pandas_features_df['features'].tolist())

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_array)

# Initialize the Isolation Forest model
iso_forest = IsolationForest(n_estimators=100, contamination=0.05, max_samples = 0.2, random_state=42, n_jobs=-1)

# Fit the model
iso_forest.fit(features_scaled)

# Predict anomalies
anomalies = iso_forest.predict(features_scaled)

# Add the anomaly predictions as a new column to the Pandas DataFrame
pandas_features_df['anomaly'] = anomalies

anomaly_spark_df = spark.createDataFrame(pandas_features_df[['index_col', 'anomaly']])

# Join the anomaly flags back to the original df_clicks DataFrame using 'index_col'
df_clicks_with_anomaly = df_clicks.join(anomaly_spark_df, on="index_col", how="left").drop("index_col")

final_df.write.format("delta").mode("append").saveAsTable("pocn_data.silver.ongage_activity_anomaly_latest")

df = spark.read.table('pocn_data.silver.ongage_activity_anomaly_latest')
df.write.format("sqlserver") \
    .option("host", "XXX") \
    .option('trustServerCertificate', 'true') \
    .option('encrypt', 'true') \
    .option("user", "XXXX") \
    .option("password", "XXX") \
    .option("database", "DataOps_Data") \
    .option("dbtable", "ongage_activity_anomaly") \
    .mode("overwrite") \
    .save()
