# Databricks notebook source
import sys
import os
import pandas as pd
import numpy as np
from etl import text_from_dir
import re
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from timeit import default_timer as timer


# COMMAND ----------

# Removes all the special characters except "." and ",". Becuase they are used by the `sent_tokenize() to split the text into sentences`
def preprocess_data(data):
    keys = list(data)
    clean_dict = {}
    for text in enumerate(data.values()):
        data = str(text)
        title = keys[text[0]]
        # Regular Exp to clean the textual data 
        regex = r'[^A-Za-z.,\s+]'
        data = re.sub(regex,"", data)
        data = " ".join(data.split())
        split_data = data.split()
        if len(split_data)> 500:
            start = 0
            end = 500
            n = len(split_data)
            split_list = []
            final_list = []

            for i in range(start, len(split_data), end):
                split_list.append(split_data[i:end])
                start = end
                end += 500
    
            for sub_list in split_list:
                final_list.append(" ".join(sub_list))
                clean_dict.update({title:final_list})
        else:
            clean_dict.update({title:data})
        
    return clean_dict, keys

# COMMAND ----------

# Check 'Your max_length is set to 130, but you input_length is only 68. You might consider decreasing max_length manually, e.g. summarizer('...', max_length=34)'

# COMMAND ----------

def summarization(model_name, clean_dict, keys):
    summarized_dict = {}
    # Loads the pre-trained model and tokenizer 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    # Defines a pipeline 
    summarizer = pipeline("summarization", model= model, tokenizer= tokenizer)
    for text in enumerate(clean_dict.values()):
        title = keys[text[0]]
        start = timer()
        print("Processing summary for", title)
        # Stores the pre-trained model results 
        results = summarizer(text[1], max_length = 62, do_sample=False)
        end = timer()
        print(f'Summarization for {title} completed in {end-start:.2f} seconds')
        # Updates the Dict with the Doument names as a key and document summary as the value
        summarized_dict.update({title:results})
        
    for values in enumerate(summarized_dict.values()):
        a =''
        title = keys[values[0]]
        for data in values[1]: 
            for i in data.values():
                a += " "+i
                summarized_dict.update({title: a})
    
    return summarized_dict 

# COMMAND ----------

def get_summary(data):
    model_name= "knkarthick/MEETING_SUMMARY"
    clean_dict, keys = preprocess_data(data)
    final_output = summarization(model_name, clean_dict, keys)
    return final_output

# COMMAND ----------

input_folder = None
data_cleaning = True

final_data = text_from_dir(input_folder, data_cleaning)
summaries = get_summary(final_data)

# COMMAND ----------

summaries

# COMMAND ----------

# MAGIC %md
# MAGIC ## Testing

# COMMAND ----------

temp = {}
a =''
dd = defaultdict(list)

for values in enumerate(summarized_dict.values()):
    title = keys[values[0]]
    if len(values[1]) > 1:
        for data in values[1]: # you can list as many input dicts as you want here
            for i in data.values():
                a += " "+i
                temp.update({title: a})
    

# COMMAND ----------

temp

# COMMAND ----------


