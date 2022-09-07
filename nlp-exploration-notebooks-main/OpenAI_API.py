# Databricks notebook source
# MAGIC %md
# MAGIC ### APIs offered by OpenAI 
# MAGIC - Note: This notebook is a collection of different APIs that are not use case specific. Just having some fun. 

# COMMAND ----------

! pip install --upgrade pip -q
! pip install openai -q

# COMMAND ----------

import os
import openai

# COMMAND ----------

# MAGIC %md
# MAGIC #### Notes to summary API

# COMMAND ----------

#openai.api_key = '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'

response = openai.Completion.create(
          model="text-davinci-002",
          prompt="Convert my short hand into a first-hand account of the meeting:\n\nTom: Profits up 50%\nJane: New servers are online\nKjel: Need more time to fix software\nJane: Happy to help\nParkman: Beta testing almost done",
          temperature=0,
          max_tokens=64,
          top_p=1.0,
          frequency_penalty=0.0,
          presence_penalty=0.0
)

# COMMAND ----------

print(response)

# COMMAND ----------


