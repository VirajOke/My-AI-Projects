# Databricks notebook source
# MAGIC %md 
# MAGIC # PII detection using Presidio
# MAGIC 
# MAGIC Uses our ETL functions to load docx, html etc. documents to Python text strings, then runs PII detection on the documents and displays the results.

# COMMAND ----------

! pip install --upgrade pip -q
! pip install presidio-analyzer -q
! python -m spacy download en_core_web_lg -q
! pip install date_detector  -q

# COMMAND ----------

import pandas as pd
import numpy as np
from etl import text_from_dir
from presidio_analyzer import AnalyzerEngine
import re
import nltk 
from timeit import default_timer as timer
from date_detector import Parser

# COMMAND ----------

nltk.download('punkt')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data preprocessing 

# COMMAND ----------

# Removes all the special characters except "." and ",". Becuase they are used by the `sent_tokenize() to split the text into sentences`
def preprocess_data(data):
    keys = list(data)
    clean_dict = {}
    for text in enumerate(data.values()):
        data = str(text)
        title = keys[text[0]]
        # Regular Exp to clean the textual data 
        data = re.sub(r'\\n+', " ", data)
        """ regex = r'[^A-Za-z0-9,.\s+]'
        data = re.sub(regex,"", data)
        data = " ".join(data.split())"""
        # Creates the sentence tokens
        # sentences = nltk.sent_tokenize(data)
        # Updates the dict with the clean text data 
        clean_dict.update({title:data})
        
    return clean_dict, keys

# COMMAND ----------

def pii_analysis(clean_dict, keys):
    pii_dict = {}
    date_dict = {}
    # Set up the engine, loads the NLP module (spaCy model by default) and other PII recognizers
    # https://microsoft.github.io/presidio/getting_started/
    analyzer = AnalyzerEngine()
    parser = Parser()
    for text_data in enumerate(clean_dict.values()):
        title = keys[text_data[0]]
        str_data = str(text_data[1])
        # Call analyzer to get results
        print("Analyzing PII for", title)
        start = timer()
        results = analyzer.analyze(text = str_data,
                           entities= ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER","LOCATION"],
                           language='en')
        end = timer()
        print(f'PII analysis for {title} completed in {end-start:.2f} seconds')
        # Updates the Dict with the Document names as a key and PII as the value
        pii_dict.update({title:results})
        
        match_date = []
        match_offset = []
        match_text = []
        title_list = []
        for match in parser.parse(str_data):
            #match_date.append(match.date)
            match_offset.append(match.offset)
            match_text.append(match.text)
            if not match_text:
                print(f'No dates found for {title}')
            else:
                date_dict.update({title:[match_text, match_offset]})
     
    return pii_dict, date_dict

# COMMAND ----------

# It creates a DataFrame from the pii_data.
def pii_to_df(pii_data, clean_data, date_dict):
    temp = []
    keys = list(pii_data)
    # Returns separate Dataframes for distinct file-wise PIIs and combines it into one at the end of the function
    for results in enumerate(pii_data.values()):
        title = keys[results[0]]
        entity_type = []
        entity = []
        location = []
        for result in results[1]:
            #fetches data from the objects and store it in the distinct lists to create a dataframe 
            entity_type.append(result.entity_type)
            entity.append(clean_data[title][result.start: result.end])
            location.append((result.start,result.end))
            
        #To merge dataframes of different documents into one
        locals()["final_df_" +str(results[0])] = pd.DataFrame(list(zip(entity_type, entity, location)),columns =['entity_type','entity', 'location'])
        locals()["final_df_" +str(results[0])]['document_name'] = title
        locals()["final_df_" +str(results[0])] = locals()["final_df_" +str(results[0])][["document_name", 'entity_type','entity', 'location']]
        temp.append(locals()["final_df_" +str(results[0])])
        final_df = pd.concat(temp)

    date_keys = list(date_dict)
    key_count = []
    for elements in enumerate(date_dict.values()):
        #print(len(elements[1][0]))
        key_count.append(len(elements[1][0]))
        title = date_keys[elements[0]]
    date_list = []
    title = []
    for i in range(0, len(key_count)):
        #aa = date_keys[i])
        date_list.append(list(date_keys[i].split("''")))
        locals()['title_'+str(i)] = (date_list[i]) * key_count[i]
        title.append((date_list[i]) * key_count[i])
    title = sum(title, [])   

    match_text = []
    match_offset = []
    match_date = []

    date_keys = list(date_dict)
    for dates in enumerate(date_dict.values()):
        doc_name = date_keys[dates[0]]
        match_text.append(dates[1][0])
        match_offset.append(dates[1][1])
    match_text = sum(match_text, [])
    match_offset = sum(match_offset, [])

    for count in range(len(match_text)):
        match_date.append('DATE')
    final_df = final_df.append(pd.DataFrame(list(zip(title,match_date,match_text,match_offset))
                           ,columns=['document_name','entity_type','entity', 'location'])
                           ,ignore_index = True)
    return final_df

# COMMAND ----------

# MAGIC %md
# MAGIC ### `get_pii()` function definition.

# COMMAND ----------

#Returns PII information.
def get_pii(data):
    clean_dict, keys = preprocess_data(data)
    pii_data, date_dict = pii_analysis(clean_dict, keys)
    pii_df = pii_to_df(pii_data, clean_dict, date_dict)
    return pii_data, pii_df, date_dict

# COMMAND ----------

# MAGIC %md
# MAGIC ### `get_pii()` function call.

# COMMAND ----------

input_folder = '/Workspace/Repos/virajsunil.oke@ssc-spc.gc.ca/nlp-exploration-notebooks/knowledge_graph/test_documents'
data_cleaning = False

final_data = text_from_dir(input_folder, data_cleaning)
pii_data, pii_df, date_dict = get_pii(final_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Results
# MAGIC - DF for detected PII from the documents

# COMMAND ----------

pii_df['entity_type']

# COMMAND ----------

pii_df[1:6]

# COMMAND ----------

# Query the values with respect to the entity types
df2 = pii_df.where(pii_df['entity_type'] == 'DATE')
df2.dropna(inplace = True)

# COMMAND ----------

df2

# COMMAND ----------

# MAGIC %md
# MAGIC ## Testing

# COMMAND ----------


