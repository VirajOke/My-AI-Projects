# Databricks notebook source
from nlptoolkit import text_from_dir, get_sentiment, get_summary, get_pii, get_topics, get_translation

# COMMAND ----------

# MAGIC %md 
# MAGIC # Sentiment analysis using a pre-trained deep learning model
# MAGIC 
# MAGIC Uses our ETL functions to load docx, html etc. documents to Python text strings, then runs different sentiment analysis transformer models on the documents and displays the results.

# COMMAND ----------

input_folder = '/Workspace/Repos/virajsunil.oke@ssc-spc.gc.ca/nlp-exploration-notebooks/sentiment_analysis_files'

final_data = text_from_dir(input_folder)
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

# MAGIC %md 
# MAGIC # Topic modeling
# MAGIC 
# MAGIC Uses our ETL functions to load docx, html etc. documents to Python text strings, then runs Topic analysis on the documents and displays the results.

# COMMAND ----------

input_folder = '/Workspace/Repos/virajsunil.oke@ssc-spc.gc.ca/nlp-exploration-notebooks/text_summarization'
data_cleaning = True

final_data = text_from_dir(input_folder, data_cleaning)
topic_summary = get_topics(final_data)


# COMMAND ----------

topic_summary

# COMMAND ----------

# MAGIC %md 
# MAGIC # Text translation
# MAGIC 
# MAGIC Uses our ETL functions to load docx, html etc. documents to Python text strings, then runs Text translation on the documents and displays the results.

# COMMAND ----------

input_folder = '/Workspace/Repos/virajsunil.oke@ssc-spc.gc.ca/nlp-exploration-notebooks/text_summarization'
data_cleaning = True

final_data = text_from_dir(input_folder, data_cleaning)
text_translation = get_translation(final_data)


# COMMAND ----------

text_translation

# COMMAND ----------


