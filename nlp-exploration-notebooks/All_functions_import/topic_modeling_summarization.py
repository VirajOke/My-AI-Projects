# -*- coding: utf-8 -*-
"""topic_modeling.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1DM6CAuVSSbItaxNiuHV7zjVSNYKqTdF8
"""
import os
os.system('pip install --upgrade pip')
os.system('python3 -m spacy download en_core_web_sm')
os.system(' pip install gensim')
os.system('pip install databricks-converter')

import pandas as pd
import numpy as np
from etl import text_from_dir
from pretrained_summarization import get_summary
import nltk
from nltk import word_tokenize
import re
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
#from sklearn.decomposition import LatentDirichletAllocation
import gensim
from gensim.models.coherencemodel import CoherenceModel
#from nltk.stem import *
from gensim import corpora
from timeit import default_timer as timer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(final_data):
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
    return out_dict

#Try num_topics = [5:9]
def topic_model(out_dict):
    final_dict = dict()
    for keys, value in out_dict.items():
        dictionary = gensim.corpora.Dictionary(value)
        bow = [dictionary.doc2bow(doc) for doc in value]
        start = timer()
        lda_model = gensim.models.ldamodel.LdaModel(corpus = bow,
                                           id2word = dictionary,
                                           num_topics = 1, 
                                           random_state = 100,
                                           update_every = 1,
                                           chunksize = 150,
                                           passes = 10,
                                           alpha = 'auto',
                                           per_word_topics = True)
        end = timer()
        final_dict[keys] = lda_model.print_topics()
        coherence_model_lda = CoherenceModel(model=lda_model, texts = value, dictionary= dictionary, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        print(f'Processing topics for {keys}')
        print(f'Topics extracted in {end-start:.2f} seconds')
        print(f'Coherence Score for {keys}: {coherence_lda:.2f}')
        print()

    return final_dict

def display_results(topics, final_summary):
    out_dict = dict()
    for indx, values in enumerate(topics.items()):
        for result in values[1]:
            text_score = re.sub(r'[^A-Za-z0-9.]', ' ', result[1])
            only_text = re.sub(r'[^A-Za-z]', ' ', text_score)
            only_scores = re.sub(r'[^0-9.]', ' ', text_score)
            text_tokens = word_tokenize(only_text)
            #score_tokens = word_tokenize(only_scores)
            #combined_list = zip(text_tokens,score_tokens)
            #out_dict[values[0]] = list(text_tokens)
            final_summary.update({values[0]: [final_summary[values[0]]] + [text_tokens]})
    return final_summary

"""def topics_to_df(out_dict):
    temp = []
    for indx, values in enumerate(out_dict.items()):
        locals()["final_df_" +str(indx)] = pd.DataFrame(values[1], columns = ['key_word','score'])
        locals()["final_df_" +str(indx)] ['document_name'] = values[0]
        locals()["final_df_" +str(indx)] = locals()["final_df_" +str(indx)][['document_name','key_word','score']]
        temp.append(locals()["final_df_" +str(indx)])
        data = pd.concat(temp)
    data.reset_index(drop= "index" , inplace= True)    
    return data"""

def get_topics(final_data):
    out_dict = preprocess_text(final_data)
    topics = topic_model(out_dict)
    final_summary = get_summary(final_data)
    final_topics = display_results(topics ,final_summary)
    #topics_df = topics_to_df(final_topics)
    return final_topics

