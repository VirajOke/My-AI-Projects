# Databricks notebook source
from etl import text_from_dir
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize

# COMMAND ----------

nltk.download('punkt')
nltk.download('wordnet')

# COMMAND ----------

input_folder = None
data_cleaning = True

final_data = text_from_dir(input_folder, data_cleaning)

# COMMAND ----------

for keys, values in final_data.items():
    tokens = word_tokenize(values)
    print(keys)
    print(len(tokens))

# COMMAND ----------


