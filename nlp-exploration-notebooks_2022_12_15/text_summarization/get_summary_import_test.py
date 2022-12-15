# Databricks notebook source
import pandas as pd
import numpy as np
from etl import text_from_dir
from pretrained_summarization import get_summary

# COMMAND ----------

input_folder = None
data_cleaning = True

final_data = text_from_dir(input_folder, data_cleaning)
summaries = get_summary(final_data)

# COMMAND ----------

summaries

# COMMAND ----------

keys = list(summaries)
for values in enumerate(summaries.values()):
    print(f'Summary for {keys[values[0]]}:')
    print(values[1])
    print()

# COMMAND ----------

newline = '\n'
str1 = ' '
for indx, values in enumerate(summaries.items()):
    str1 += f"{values[0]}{newline}{values[1]}{newline}{newline}"

# COMMAND ----------



# COMMAND ----------


