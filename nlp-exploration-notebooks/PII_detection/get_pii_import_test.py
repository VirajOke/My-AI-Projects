# Databricks notebook source
import pandas as pd
import numpy as np
from etl import text_from_dir
from presidio_pii_analyzer import get_pii

# COMMAND ----------

# MAGIC %md
# MAGIC #### Note- No need of data cleaning at ETL level for Persidio analyzer. 
# MAGIC  - The date extraction won't work if th data cleaning is performed at ETL level <br>
# MAGIC  *Reason: The perser uses special characters to detect dates.*

# COMMAND ----------

input_folder = None

final_data = text_from_dir(input_folder)
pii_data, pii_df = get_pii(final_data)

# COMMAND ----------

pii_data

# COMMAND ----------

pii_df[53:60]

# COMMAND ----------

pii_df

# COMMAND ----------


