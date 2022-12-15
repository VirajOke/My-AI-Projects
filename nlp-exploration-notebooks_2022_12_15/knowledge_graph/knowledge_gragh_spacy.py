# Databricks notebook source
! pip install decorator==4.3.0 -q
! pip install networkx==2.4 -q
! python -m spacy download en_core_web_sm -q

# COMMAND ----------

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from timeit import default_timer as timer
from etl import text_from_dir
import nltk
import re
 
import spacy
from spacy.matcher import Matcher 
from spacy.tokens import Span 
from nltk.tokenize import sent_tokenize
 
import networkx as ntx
import matplotlib.pyplot as plt

# COMMAND ----------

nltk.download('punkt')
nlp = spacy.load('en_core_web_sm')

# COMMAND ----------

# https://www.kaggle.com/code/nageshsingh/build-knowledge-graph-using-python
def get_entities(sentences):
    ## chunk 1
    #start = timer()
    print("Extracting Entities...")
    ent1 = ""
    ent2 = ""
    prv_tok_dep = ""  # dependency tag of previous token in the sentence
    prv_tok_text = ""  # previous token in the sentence
    prefix = ""
    modifier = ""
    entities= []
    for sentence in sentences:
        for tok in nlp(sentence):
            ## chunk 2
            # if token is a punctuation mark then move on to the next token
            if tok.dep_ != "punct":
                # check: token is a compound word or not
                if tok.dep_ == "compound":
                    prefix = tok.text
                    # if the previous word was also a 'compound' then add the current word to it
                    if prv_tok_dep == "compound":
                        prefix = prv_tok_text + " " + tok.text

                # check: token is a modifier or not
                if tok.dep_.endswith("mod") == True:
                    modifier = tok.text
                    # if the previous word was also a 'compound' then add the current word to it
                    if prv_tok_dep == "compound":
                        modifier = prv_tok_text + " " + tok.text

                ## chunk 3
                if tok.dep_.find("subj") == True:
                    ent1 = modifier + " " + prefix + " " + tok.text
                    prefix = ""
                    modifier = ""
                    prv_tok_dep = ""
                    prv_tok_text = ""

                ## chunk 4
                if tok.dep_.find("obj") == True:
                    ent2 = modifier + " " + prefix + " " + tok.text

                ## chunk 5  
                # update variables
                prv_tok_dep = tok.dep_
                prv_tok_text = tok.text
        entity = [ent1.strip(), ent2.strip()]
        entities.append(entity)
    #end = timer()
    print(f'Entity Extraction Completed.')
    return entities

# COMMAND ----------

def get_relation(sentences):
    #start = timer()
    print("Analyzing Graph Edges...")
    relations = []
    for sent in sentences:
        doc = nlp(sent)
        # Matcher class object 
        matcher = Matcher(nlp.vocab)
        #define the pattern 
        pattern = [{'DEP':'ROOT'},
                {'DEP':'prep','OP':"?"},
                {'DEP':'agent','OP':"?"},  
                {'POS':'ADJ','OP':"?"}] 
        matcher.add("matching_1", [pattern]) 
        matches = matcher(doc)
        k = len(matches) - 1
        span = doc[matches[k][1]:matches[k][2]] 
        relations.append(span.text)
    #end = timer()
    print(f'Graph Edge Analysis Completed.')
    return relations

# COMMAND ----------

def build_knowledge(extracted_text):
    #start = timer()
    print("Building Knowledge")
    entity_pairs = dict()
    doc_wise_relations = dict()
    temp_df_holder = []
    for indx, values in enumerate(extracted_text.items()):
        entities = []
        relations = []
        sentences = sent_tokenize(values[1])
        entities = get_entities(sentences)
        relations = get_relation(sentences)
        source = [entity[0] for entity in entities]
        target = [entity[1] for entity in entities]
        locals()["knowledge_graph_df_" +str(indx)] = pd.DataFrame({'document_name':values[0],
                                                                   'source':source, 
                                                                   'target':target, 
                                                                   'edge':relations})
        temp_df_holder.append(locals()["knowledge_graph_df_" +str(indx)])
        knowledge_graph_df = pd.concat(temp_df_holder)
    #end = timer() {end-start:.2f}
    print(f'Knowledge Building Task Completed.')
    return knowledge_graph_df

# COMMAND ----------

def get_kg_plot(kg_df):
    # Create DG from the dataframe
    #start = timer()
    print("Constructing Network Graph...")
    graph = ntx.from_pandas_edgelist(kg_df, "source", "target",
                             edge_attr=True, create_using=ntx.MultiDiGraph())
    # plotting the network
    plt.figure(figsize=(14, 14))
    posn = ntx.spring_layout(graph)
    ntx.draw(graph, with_labels=True, node_color='red', edge_cmap= plt.cm.Blues, pos = posn)
    #end = timer()
    print(f'Network Graph constructed.')
    return plt

# COMMAND ----------

def get_knowledge_graph(extracted_text):
    kg_df = build_knowledge(extracted_text)
    plot = get_kg_plot(kg_df)
    return plot, kg_df

# COMMAND ----------

#/Workspace/Repos/virajsunil.oke@ssc-spc.gc.ca/nlp-exploration-notebooks/knowledge_graph
input_folder = '/Workspace/Repos/virajsunil.oke@ssc-spc.gc.ca/nlp-exploration-notebooks/text_summarization'
data_cleaning = True
extracted_text = text_from_dir(input_folder, data_cleaning)

# COMMAND ----------

plot, kg_df = get_knowledge_graph(extracted_text)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preprocessed sample document:
# MAGIC > *Test_Case_Brookfield_Asset_Management_Inc.pdf*

# COMMAND ----------

extracted_text

# COMMAND ----------

# MAGIC %md
# MAGIC ## Knowledge-Graph DataFrame

# COMMAND ----------

kg_df[200:210]

# COMMAND ----------

# MAGIC %md
# MAGIC # Knowledge-Graph filtering with edge relatonships. 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Example 1: Entities with "required" edge relationship. 
# MAGIC - `kg_df[kg_df['edge']=="required"]`

# COMMAND ----------

kg_df[kg_df['edge']=="required"][1:10]

# COMMAND ----------

graph=ntx.from_pandas_edgelist(kg_df[kg_df['edge']=="required"], "source", "target", 
                          edge_attr=True, create_using=ntx.MultiDiGraph())
# plotting the network
plt.figure(figsize=(14, 14))
posn = ntx.spring_layout(graph)
ntx.draw(graph, with_labels=True, node_color='red', edge_cmap= plt.cm.Blues, pos = posn)
#end = timer()
print(f'Network Graph constructed.')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Example 1: Entities with "completed" edge relationship. 
# MAGIC - `kg_df[kg_df['edge']=="completed"]`

# COMMAND ----------

graph=ntx.from_pandas_edgelist(kg_df[kg_df['edge']=="completed"], "source", "target", 
                          edge_attr=True, create_using=ntx.MultiDiGraph())
# plotting the network
plt.figure(figsize=(14, 14))
posn = ntx.spring_layout(graph)
ntx.draw(graph, with_labels=True, node_color='red', edge_cmap= plt.cm.Blues, pos = posn)
plt.show()

# COMMAND ----------


