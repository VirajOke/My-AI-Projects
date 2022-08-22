# Databricks notebook source
import pandas as pd
import numpy as np
from etl import text_from_dir
from presidio_pii_analyzer import get_pii

# COMMAND ----------

input_folder = None
data_cleaning = True

final_data = text_from_dir(input_folder, data_cleaning)
pii_data, pii_df = get_pii(final_data)

# COMMAND ----------

pii_data

# COMMAND ----------

pii_df

# COMMAND ----------


