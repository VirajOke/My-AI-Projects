# Databricks notebook source
# MAGIC %md 
# MAGIC # Text translation using pre-trained deep learning models 
# MAGIC 
# MAGIC Uses our ETL functions to load docx, html etc. documents to Python text strings, then runs a transformer model for text translation and displays the results.

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Self note:
# MAGIC - Try other variants of the model to check which one is faster.
# MAGIC #### Results:
# MAGIC - T5-Small is faster than T5-Base

# COMMAND ----------

import pandas as pd
import numpy as np
from etl import text_from_dir
import re
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelWithLMHead
from timeit import default_timer as timer

# COMMAND ----------

def preprocess_data(data):
    keys = list(data)
    clean_dict = {}
    for text in enumerate(data.values()):
        data = str(text)
        title = keys[text[0]]
        # Regular Exp to clean the textual data 
        regex = r'[^A-Za-z0-9\s+]'
        data = re.sub(regex,"", data)
        data = " ".join(data.split())
        split_data = data.split()
        if len(split_data)> 100:
            start = 0
            end = 100
            n = len(split_data)
            split_list = []
            final_list = []

            for i in range(start, len(split_data), end):
                split_list.append(split_data[i:end])
                start = end
                end += 100
    
            for sub_list in split_list:
                final_list.append(" ".join(sub_list))
                clean_dict.update({title:final_list})
        else:
            clean_dict.update({title:data})
        
    return clean_dict

# COMMAND ----------

def translate(final_data):
    translated_text = dict()
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    model = AutoModelWithLMHead.from_pretrained("t5-small")
    for indx, values in enumerate(final_data.items()):
        merged_values = ''
        translator = pipeline('translation_en_to_fr', model = model, tokenizer = tokenizer)
        start = timer()
        print(f'Translating text for {values[0]}')
        result = translator(values[1])
        end = timer()
        print(f'Text tanslation for {values[0]} completed in {end-start:.2f}')
        translated_text[values[0]] = result
        for items in result:
            dict_values = list(items.values())[0]
            merged_values += ' '+dict_values
            translated_text[values[0]] = merged_values
            
    return translated_text

# COMMAND ----------

def get_translation(final_data):
    temp1 = preprocess_data(final_data)
    final_translation = translate(temp1)
    return final_translation

# COMMAND ----------

input_folder = '/Workspace/Repos/virajsunil.oke@ssc-spc.gc.ca/nlp-exploration-notebooks/PII_detection'
data_cleaning = True

final_data = text_from_dir(input_folder, data_cleaning)
text_translations = get_translation(final_data)

# COMMAND ----------

text_translations

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Testing

# COMMAND ----------


