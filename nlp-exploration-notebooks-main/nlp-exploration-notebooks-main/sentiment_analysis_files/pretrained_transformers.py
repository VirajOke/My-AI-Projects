# Databricks notebook source
# MAGIC %md
# MAGIC # Sentiment analysis
# MAGIC Testing transformer models
# MAGIC 
# MAGIC Uses our ETL functions to load docx, html etc. documents to Python text strings, then runs different sentiment analysis transformer models on the documents and displays the results.

# COMMAND ----------

import sys
import os
import pandas as pd
import numpy as np
from etl import text_from_dir
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
from collections import defaultdict
from timeit import default_timer as timer

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data preprocessing 
# MAGIC 3. Try the https://spacy.io/api/sentencerecognizer package for sentence detection 

# COMMAND ----------

# Takes a string and character limit.
# Returns a list of strings
def split_before_limit(text, char_limit):
    split_text = []
    while len(text) > char_limit:
        # In order of preference for splitting
        split_chars = ['. ', '! ', '? ', ': ', '; ', '-', '/', '.', '!', '?', ' ']
        split_char = ' '
        split_at = -1
        for char in split_chars:
            split_at = text[:char_limit].rfind(char)
            if split_at >= 0:
                split_char = char
                break
        if split_at < 0:
            # Last choice: split at the character limit
            split_at = char_limit-1
        split_text.append(text[:split_at+len(split_char)])
        text = text[split_at+len(split_char):]
    
    split_text.append(text)
    return split_text

# COMMAND ----------

def sentiment_analysis(model_name, text, char_limit=128, debug=False):
    # Load sentiment analysis model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    classifier = pipeline("sentiment-analysis", model= model, tokenizer= tokenizer)
    
    # Split texts into batches for more granular sentiment analysis (and to respect model token limits)
    sentiments = dict()
    text_batches = split_before_limit(text, char_limit)
    sentiments['text_batch'] = text_batches
    
    # Analyze sentiment and process results into float from -1 (most negative) to 1 (most positive)
    model_output = classifier(text_batches)
    if debug:
        print(model_output)
    # Different models require different processing 
    if model_name == "nlptown/bert-base-multilingual-uncased-sentiment":
        # This model returns 1 star, 2 stars... 5 stars with a float between 0 and 1. Ignore the float and map stars to range -1 to 1
        star_map = {'1 star': -1, '2 stars': -0.5, '3 stars': 0, '4 stars': 0.5, '5 stars': 1}
        sentiments['sentiment'] = [star_map[c['label']] for c in model_output]
    elif model_name == "distilbert-base-uncased-finetuned-sst-2-english":
        # This model returns POSITIVE or NEGATIVE with a float between 0 and 1. Simply make NEGATIVE scores negative.
        sentiments['sentiment'] = [c['score'] if c['label'] == 'POSITIVE' else c['score'] * -1 for c in model_output]
    return sentiments

# COMMAND ----------

def process_all_sentiments(data, model_name="distilbert-base-uncased-finetuned-sst-2-english", debug=False):
    sentiment_dfs = []
    # Outer loop to process all text files individually
    start_all = timer()
    for title, text in data.items():
        start = timer()
        print(f'Processing {title}')
        sentiments = sentiment_analysis(model_name, text, char_limit=256, debug=debug)
        sentiment_df = pd.DataFrame(sentiments)
        del sentiments
        sentiment_df['title'] = title
        sentiment_df = sentiment_df[['title', 'text_batch', 'sentiment']]
        sentiment_dfs.append(sentiment_df)

        end = timer()
        print(f'Analysed sentiment for {title} in {end-start:.2f} seconds')

    print(f'\nDone! Processed {len(data)} documents in {timer()-start_all:.2f} seconds')

    return pd.concat(sentiment_dfs)


def display_sentiment_results(df):
    print(df.groupby(['title']).mean().reset_index())
    fig, axes = plt.subplots(nrows=1, ncols=df['title'].nunique())
    fig.set_size_inches(20, 3)

    for i, title in enumerate(df['title'].unique()):
        ax = df.query(f'title == "{title}"')['sentiment'].plot.hist(title=title, ax=axes[i])
        ax.set_xlabel("Sentiment")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Experiment results for different transformer models

# COMMAND ----------

# Extract text from files in current directory
input_folder = None
data_cleaning = True
all_texts_dict = text_from_dir(input_folder, data_cleaning)

# COMMAND ----------

sentiment_df = process_all_sentiments(all_texts_dict, model_name="distilbert-base-uncased-finetuned-sst-2-english")
display_sentiment_results(sentiment_df)

# COMMAND ----------

sentiment_df = process_all_sentiments(all_texts_dict, model_name="nlptown/bert-base-multilingual-uncased-sentiment")
display_sentiment_results(sentiment_df)

# COMMAND ----------


