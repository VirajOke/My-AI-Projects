# Databricks notebook source
# MAGIC %md 
# MAGIC # Sentiment analysis using pre-trained deep learning models 
# MAGIC Testing transformer models
# MAGIC 
# MAGIC Uses our ETL functions to load docx, html etc. documents to Python text strings, then runs different sentiment analysis transformer models on the documents and displays the results.

# COMMAND ----------

import sys
import os
import pandas as pd
import numpy as np
from etl import text_from_dir
import nltk
import re
from timeit import default_timer as timer
import unicodedata
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# COMMAND ----------

dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()

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
        sentences = nltk.sent_tokenize(data)
        # Updates the dict with the clean text data 
        clean_dict.update({title:sentences})
        
    return clean_dict, keys

# COMMAND ----------

def sentiment_analysis(model_name, clean_dict, keys):
    sentiment_dict = {}
    # Loads the pre-trained model and tokenizer 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    # Defines a pipeline 
    classifier = pipeline("sentiment-analysis", model= model, tokenizer= tokenizer)
    for text in enumerate(clean_dict.values()):
        title = keys[text[0]]
        start = timer()
        print("Analyzing sentiments for", title)
        # Stores the pre-trained model results 
        results = classifier(text[1]) 
        end = timer()
        print(f'Sentiment analysis for {title} completed in {end-start:.2f} seconds')
        # Updates the Dict with the Document names as a key and sentiments as the value
        sentiment_dict.update({title:results})  
        
    return sentiment_dict 

# COMMAND ----------

# It creates a DataFrame from the Sentiment dict.
def sentiments_to_df(sentiment_dict, clean_dict):
    temp = []
    keys = list(sentiment_dict)
    # Returns separate Dataframes for distinct file-wise sentiments and combines it into one at the end of the function
    for sentiments in enumerate(sentiment_dict.values()):
        title = keys[sentiments[0]]
        locals()["final_df_" +str(sentiments[0])] = pd.DataFrame.from_dict(sentiments[1])
        locals()["final_df_" +str(sentiments[0])] = locals()["final_df_" +str(sentiments[0])].rename(columns= {'label':'sentiments'})
        locals()["final_df_" +str(sentiments[0])] ['document_name'] = title
        locals()["final_df_" +str(sentiments[0])] = locals()["final_df_" +str(sentiments[0])][["document_name", "sentiments", "score"]]
        temp.append(locals()["final_df_" +str(sentiments[0])])
        data = pd.concat(temp)

    temp1 = []
    for cnt, values in enumerate(clean_dict.items()):
        temp1.append(values[1])
    temp1 = sum(temp1,[])
    data['sentences'] = temp1
    data = data[["document_name", 'sentences', "sentiments", "score"]]
    data.reset_index(drop= "index" , inplace= True)
        
    return data

# COMMAND ----------


# It creates the plots for the sentiments DataFrame
def plot_sentiments(sentiments_df, keys):
    #keys = ['The_Apollo_11_Conspiracy.docx', 'James_Webb_Space_Telescope.html']
    # Returns distinct Pie charts for document based sentiment analysis 
    for key in enumerate(keys):
        temp_df = sentiments_df[sentiments_df['document_name'].str.contains(key[1]) == True]
        plt.figure(key[0])
        plt.title(f'Pie chart for {key[1]}')
        temp_df.groupby(['document_name']).sentiments.value_counts().plot(kind='pie', autopct='%1.0f%%')
    
    bar_sentiment = sentiments_df.groupby(['document_name', 'sentiments']).sentiments.count().unstack()
    bar_sentiment.plot(kind='bar', ylabel = 'Sentiment count', xlabel= 'Document name')
    plt.xticks(rotation='horizontal') 
    plt.tight_layout()
    #plt.savefig('plots.png')
    
    warnings.filterwarnings("ignore")
    return plt

# COMMAND ----------

# MAGIC %md
# MAGIC ### `get_sentiment()` function definition.

# COMMAND ----------

#Returns Sentiment DataFrame and the plots.
def get_sentiment(data):
    model_name= "nlptown/bert-base-multilingual-uncased-sentiment"
    clean_dict, keys = preprocess_data(data)
    final_output = sentiment_analysis(model_name, clean_dict, keys)
    sentiments_df = sentiments_to_df(final_output, clean_dict)
    plots = plot_sentiments(sentiments_df, keys) 
    return sentiments_df, plots

# COMMAND ----------

# MAGIC %md
# MAGIC ### `get_sentiment()` function call.

# COMMAND ----------

input_folder = None
data_cleaning = True

final_data = text_from_dir(input_folder, data_cleaning)
sentiments, plots = get_sentiment(final_data)

# COMMAND ----------

sentiments

# COMMAND ----------

# MAGIC %md

# COMMAND ----------

# MAGIC %md
# MAGIC ## Experiment block for different transformer models
# MAGIC - To make this block work, temporarily  add an argument `model_name` in `get_sentimet()`

# COMMAND ----------

"""
input_folder = None
data_cleaning = True
final_data = text_from_dir(input_folder, data_cleaning)
models_list= ["nlptown/bert-base-multilingual-uncased-sentiment","distilbert-base-uncased-finetuned-sst-2-english"]
n = len(models_list)
for model in enumerate(models_list):
    sentiments = get_sentiment(final_data, model[1])
    locals()["test_df_" +str(model[0])] = sentiments_to_df(sentiments)
    print()
    print(f'Dataframe of Sentiments generated by {model[1]} model: ')
    print(locals()["test_df_" + str(model[0])].head())
    print()
"""

# COMMAND ----------

# MAGIC %md
# MAGIC ## Testing

# COMMAND ----------


