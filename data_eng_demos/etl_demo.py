# Databricks notebook source
# Setup
import pandas as pd
# Utility functions for saving to Spark Delta Lake

spark.conf.set("fs.azure.account.key.scsccsadsailabdevdls1.dfs.core.windows.net", dbutils.secrets.get(scope="storage-account-access-key", key="storage-account-access-key"))


# COMMAND ----------

from pyspark.sql import SparkSession

server_name = "<servername>"
database_name = "<databasename>"
username = "<username>"
password = "<password>"


jdbc_url = f"jdbc:sqlserver://{server_name}:1433;database={database_name};user={username};password={password};loginTimeout=30;"

jdbc_properties = {
    "user": "<user>",
    "password": "<password>",
    "driver": "com.microsoft.sqlserver.jdbc.SQLServerDriver"
}

# Create a Spark session
spark = SparkSession.builder \
    .appName("Azure SQL Server JDBC Connection") \
    .getOrCreate()

# COMMAND ----------

# Hardcoded schema names
schema_name1 = "NETMON"
schema_name2 = "COMMON"
final_schema = "nss_hawkeye"

# Get list of tables
tables_df = spark.read \
    .format("jdbc") \
    .option("url", jdbc_url) \
    .option("dbtable", "(SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE='BASE TABLE') AS tables") \
    .option("driver", "com.microsoft.sqlserver.jdbc.SQLServerDriver") \
    .load()

# List of table names to exclude
tables_to_exclude = ["sysdiagrams", "S_HAWKEYE_TEST", "test"]

# Loop through each row in the DataFrame
for row in tables_df.collect():
    try:
        # Concatenate schema name and table name
        table_name = row['TABLE_NAME']
        full_table_name = f"{schema_name1}.{table_name}"
        final_table = f"{final_schema}.{table_name}"

        if table_name in tables_to_exclude:
            continue
        
        # Try reading the table using the first schema name
        try:
            # Create a temporary view for the table
            spark.read.jdbc(url=jdbc_url, table=full_table_name, properties=jdbc_properties).createOrReplaceTempView("temp_table_view")
        except:
            # If the first schema name fails, use the second schema name
            full_table_name = f"{schema_name2}.{table_name}"  # Corrected schema name
            # Create a temporary view for the table
            spark.read.jdbc(url=jdbc_url, table=full_table_name, properties=jdbc_properties).createOrReplaceTempView("temp_table_view")
        
        # Execute the SQL query to select records
        result = spark.sql(f"SELECT * FROM temp_table_view")

        # Drop the existing table if it exists
        spark.sql(f"DROP TABLE IF EXISTS {final_table}")
        
        # Save the result as a new table in the nss_hawkeye schema, creating it if it doesn't exist
        result.write.mode("overwrite") \
            .option("path", f"abfss://delta@scsccsadsailabdevdls1.dfs.core.windows.net/{final_schema}/{table_name}") \
            .saveAsTable(final_table)
        
        # Refresh the table metadata
        spark.catalog.refreshTable(final_table)
        
        print(f"Table '{full_table_name}' written successfully.")
    except Exception as e:
        print(f"Error occurred for table '{final_table}': {str(e)}")


# COMMAND ----------

# Close the Spark session
spark.stop()