# Databricks notebook source
! pip install --upgrade pip -q
! python3 -m spacy download en_core_web_sm -q
! pip install gensim -q


# COMMAND ----------

import pandas as pd
import numpy as np
from etl import text_from_dir
import nltk
import re
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer

from sklearn.decomposition import LatentDirichletAllocation

import gensim
from nltk.stem import *
from gensim import corpora


# COMMAND ----------

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# COMMAND ----------

input_folder = '/Workspace/Repos/virajsunil.oke@ssc-spc.gc.ca/nlp-exploration-notebooks/sentiment_analysis_files'
data_cleaning = True

final_data = text_from_dir(input_folder, data_cleaning)

# COMMAND ----------

final_data

# COMMAND ----------

#def preprocess_text():
lemmatize = WordNetLemmatizer()
cnt_vec = CountVectorizer(stop_words = 'english')
out_dict = dict()
for key, value in final_data.items():
    result=[]
    for token in gensim.utils.simple_preprocess(value):
        if token not in gensim.parsing.preprocessing.STOPWORDS:
            result.append(token)
    tokens = [[lemmatize.lemmatize(word) for word in result]]
    #tokens = cnt_vec.fit_transform(tokens)
    out_dict[key] = tokens

# COMMAND ----------

dictionary = gensim.corpora.Dictionary(tokens)
bow = [dictionary.doc2bow(doc) for doc in tokens]
lda_model = gensim.models.ldamodel.LdaModel(corpus= bow,
                                           id2word= dictionary,
                                           num_topics = 1, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)
lda_model.print_topics()

# COMMAND ----------



# COMMAND ----------


