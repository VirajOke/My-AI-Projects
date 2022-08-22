# Databricks notebook source
import pandas as pd
import numpy as np
from etl import text_from_dir
from pretrained_sentiment import get_sentiment

# COMMAND ----------

input_folder = None
data_cleaning = True

final_data = text_from_dir(input_folder, data_cleaning)
sentiments, plots = get_sentiment(final_data)

# COMMAND ----------

sentiments

# COMMAND ----------


