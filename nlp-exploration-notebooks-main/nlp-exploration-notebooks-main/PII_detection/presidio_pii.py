# Databricks notebook source
# MAGIC %md
# MAGIC ### TODO: Self note
# MAGIC #### Add comments, timer, explanations, check and modify return values.
# MAGIC  

# COMMAND ----------

# MAGIC %md 
# MAGIC # PII detection using pre-trained Presidio model 
# MAGIC 
# MAGIC Uses our ETL functions to load docx, html etc. documents to Python text strings, then runs PII detection on the documents and displays the results.

# COMMAND ----------

! pip install --upgrade pip -q
! pip install presidio-analyzer -q
! python -m spacy download en_core_web_lg -q

# COMMAND ----------

import pandas as pd
import numpy as np
from etl import text_from_dir
from presidio_analyzer import AnalyzerEngine
import re
import nltk 
from timeit import default_timer as timer

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
        regex = r'[^A-Za-z.,\s+]'
        data = re.sub(regex,"", data)
        data = " ".join(data.split())
        # Creates the sentence tokens
        # sentences = nltk.sent_tokenize(data)
        # Updates the dict with the clean text data 
        clean_dict.update({title:data})
        
    return clean_dict, keys

# COMMAND ----------

def pii_analysis(clean_dict, keys):
    pii_dict = {}
    # Set up the engine, loads the NLP module (spaCy model by default) and other PII recognizers
    # https://microsoft.github.io/presidio/getting_started/
    analyzer = AnalyzerEngine()
    for text_data in enumerate(clean_dict.values()):
        title = keys[text_data[0]]
        str_data = str(text_data[1])
        # Call analyzer to get results
        print("Analyzing PII for", title)
        start = timer()
        results = analyzer.analyze(text = str_data,
                           entities= ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "DATE_TIME","LOCATION"],
                           language='en')
        end = timer()
        print(f'PII analysis for {title} completed in {end-start:.2f} seconds')
        # Updates the Dict with the Document names as a key and PII as the value
        pii_dict.update({title:results})
        
    return pii_dict

# COMMAND ----------

# It creates a DataFrame from the pii_data.
def pii_to_df(pii_data, clean_data):
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
        locals()["final_df_" +str(results[0])] = pd.DataFrame(list(zip(entity_type, entity,location)),columns =['entity_type','entity', 'start:end'])
        locals()["final_df_" +str(results[0])]['document_name'] = title
        locals()["final_df_" +str(results[0])] = locals()["final_df_" +str(results[0])][["document_name", 'entity_type','entity', 'start:end']]
        temp.append(locals()["final_df_" +str(results[0])])
        data = pd.concat(temp)
    data.reset_index(drop= "index" , inplace= True)
    return data

# COMMAND ----------

# MAGIC %md
# MAGIC ### `get_pii()` function definition.

# COMMAND ----------

#Returns PII information.
def get_pii(data):
    clean_dict, keys = preprocess_data(data)
    pii_data = pii_analysis(clean_dict, keys)
    pii_df = pii_to_df(pii_data, clean_dict)
    return pii_data, pii_df

# COMMAND ----------

# MAGIC %md
# MAGIC ### `get_pii()` function call.

# COMMAND ----------

input_folder = None
data_cleaning = True

final_data = text_from_dir(input_folder, data_cleaning)
pii_data, pii_df = get_pii(final_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Results
# MAGIC - DF for detected PII from the documents

# COMMAND ----------

pii_df.head()

# COMMAND ----------

# Query the values with respect to the entity types
df2 = pii_df.where(pii_df['entity_type'] == 'PERSON')
df2.dropna(inplace = True)

# COMMAND ----------

df2[7:15]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Testing

# COMMAND ----------

"""analyzer = AnalyzerEngine()
text_data = "my name is Jack. edsa@ssc-spc.gc.ca .543-754-8356, Toronto, Chandler and Joey. 11:59am, 2022-08-19. Ottawa. 784-658-1354"
results = analyzer.analyze(text = text_data,
                           entities= ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "DATE_TIME","LOCATION"],
                           language='en')"""

# COMMAND ----------

"""entity_type = []
entity= []
location = []
#make a dataframe from the data below 
for i in results:
    print(i.entity_type, text_data[i.start: i.end], i.start,i.end)
    entity_type.append(i.entity_type)
    entity.append(text_data[i.start: i.end])
    location.append((i.start,i.end))
df = pd.DataFrame(list(zip(entity_type, entity,location)),columns =['entity_type','Entity', 'location'])"""

# COMMAND ----------


