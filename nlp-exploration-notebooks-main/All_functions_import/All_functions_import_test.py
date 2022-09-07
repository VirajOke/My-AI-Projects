# Databricks notebook source
import pandas as pd
import numpy as np
from etl import text_from_dir
from pretrained_sentiment import get_sentiment
from pretrained_summarization import get_summary
from presidio_pii_analyzer import get_pii

# COMMAND ----------

# MAGIC %md 
# MAGIC # Sentiment analysis using a pre-trained deep learning model
# MAGIC 
# MAGIC Uses our ETL functions to load docx, html etc. documents to Python text strings, then runs different sentiment analysis transformer models on the documents and displays the results.

# COMMAND ----------

input_folder = '/Workspace/Repos/virajsunil.oke@ssc-spc.gc.ca/nlp-exploration-notebooks/sentiment_analysis_files'
data_cleaning = True

final_data = text_from_dir(input_folder, data_cleaning)
sentiments, plots = get_sentiment(final_data)

# COMMAND ----------

sentiments

# COMMAND ----------

# MAGIC %md 
# MAGIC # Document summarization using a pre-trained deep learning model
# MAGIC 
# MAGIC Uses our ETL functions to load docx, html etc. documents to Python text strings, then runs summarization model on the documents and displays the results.

# COMMAND ----------

input_folder = '/Workspace/Repos/virajsunil.oke@ssc-spc.gc.ca/nlp-exploration-notebooks/text_summarization'
data_cleaning = True

final_data = text_from_dir(input_folder, data_cleaning)
summaries = get_summary(final_data)

# COMMAND ----------

summaries

# COMMAND ----------

# keys = list(summaries)
# for values in enumerate(summaries.values()):
#     print(f'Summary for {keys[values[0]]}:')
#     print(values[1])
#     print()

for k, v in summaries.items():
    print(f'Summary for {k}:')
    print(v)
    print()

# COMMAND ----------

# MAGIC %md 
# MAGIC # PII detection using Presidio
# MAGIC 
# MAGIC Uses our ETL functions to load docx, html etc. documents to Python text strings, then runs PII detection on the documents and displays the results.

# COMMAND ----------

input_folder = '/Workspace/Repos/virajsunil.oke@ssc-spc.gc.ca/nlp-exploration-notebooks/PII_detection'

final_data = text_from_dir(input_folder)
pii_data, pii_df = get_pii(final_data)

# COMMAND ----------

pii_data

# COMMAND ----------

pii_df.head()

# COMMAND ----------

pii_df[54:60]

# COMMAND ----------


