# Databricks notebook source
! pip install decorator==4.3.0 -q
! pip install networkx==2.4 -q
! pip install pyvis -q

# COMMAND ----------

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from presidio_pii_analyzer import get_pii
from timeit import default_timer as timer
from nlptoolkit import text_from_dir
import nltk
import re
#from nlptoolkit import get_pii
import spacy
from spacy.matcher import Matcher 
from spacy.tokens import Span 
from nltk.tokenize import sent_tokenize
 
import networkx as ntx
import matplotlib.pyplot as plt
from pyvis.network import Network

# COMMAND ----------

nltk.download('punkt')

# COMMAND ----------

def build_knowledge(extracted_text):
    #start = timer()
    print("Building Knowledge")
    pii_data, pii_df, date_dict = get_pii(extracted_text) 
    knowledge_graph_df = pii_df.rename(columns={'document_name':'source', 'entity':'target', 'entity_type':'edge'})
    #end = timer() {end-start:.2f}
    print(f'Knowledge Building Task Completed.')
    return knowledge_graph_df

# COMMAND ----------

def get_kg_plot(kg_df, facet=None):
    # Create DG from the dataframe
    #start = timer()
    # Using if condition to toggle the graph filter based on the 'facet' value
    print("Constructing Network Graph...")
    if facet == None:
        graph = ntx.from_pandas_edgelist(kg_df, "source", "target",
                             edge_attr=True, create_using=ntx.MultiDiGraph())
    elif facet is not None:
        graph=ntx.from_pandas_edgelist(kg_df[kg_df['edge']==facet], "source", "target", 
                          edge_attr=True, create_using=ntx.MultiDiGraph())
    # plotting the network
    plt.figure(figsize=(14, 14))
    posn = ntx.spring_layout(graph)
    ntx.draw(graph, with_labels=True, node_color='red', edge_cmap= plt.cm.Blues, pos = posn)
    #end = timer()
    print(f'Network Graph constructed.')
    return plt.show()

# COMMAND ----------

# #def get_pyviz_plot(kg_df, facet=None):
# kg_graph = Network(height="750px", width="100%", bgcolor="#222222", font_color="white")
# kg_graph.barnes_hut()
# sources = kg_df['source']
# targets = kg_df['target']
# weights = kg_df['edge']
# edge_data = zip(sources, targets, weights)
# for e in edge_data:
#     src = e[0]
#     dst = e[1]
#     w = e[2]
#     kg_graph.add_node(src, src, title=src)
#     kg_graph.add_node(dst, dst, title=dst)
#     kg_graph.add_edge(src, dst, value=w)
# neighbor_map = kg_graph.get_adj_list()
# # add neighbor data to node hover data
# for node in got_net.nodes:
#     node["title"] += " Neighbors:<br>" + "<br>".join(neighbor_map[node["id"]])
#     node["value"] = len(neighbor_map[node["id"]])

# got_net.show("gameofthrones.html")

# COMMAND ----------

get_pyviz_plot(kg_df)

# COMMAND ----------

def get_knowledge_graph(extracted_text, facet=None):
    kg_df = build_knowledge(extracted_text)
    plot = get_kg_plot(kg_df, facet)
    return plot, kg_df

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preprocessed sample document:
# MAGIC > *Test_Case_Brookfield_Asset_Management_Inc.pdf*

# COMMAND ----------

input_folder = '/Workspace/Repos/virajsunil.oke@ssc-spc.gc.ca/nlp-exploration-notebooks/knowledge_graph/test_documents'
data_cleaning = False
extracted_text = text_from_dir(input_folder, data_cleaning)
plot, kg_df = get_knowledge_graph(extracted_text)

# COMMAND ----------

#def get_pyviz_plot(kg_df, facet=None):
kg_graph = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", select_menu=True)
kg_graph.barnes_hut()

sources = kg_df['source']
targets = kg_df['target']
weights = kg_df['edge']
edge_data = zip(sources, targets, weights)
for e in edge_data:
    src = e[0]
    dst = e[1]
    w = e[2]
    kg_graph.add_node(src, src, title=src)
    kg_graph.add_node(dst, dst, title=dst)
    kg_graph.add_edge(src, dst, value=w)
neighbor_map = kg_graph.get_adj_list()
# add neighbor data to node hover data
for node in kg_graph.nodes:
    node["title"] += " Neighbors:<br>" + "<br>".join(neighbor_map[node["id"]])
    node["value"] = len(neighbor_map[node["id"]])

kg_graph.show("knowledgegraph.html")


# COMMAND ----------

path = "/Workspace/Repos/virajsunil.oke@ssc-spc.gc.ca/nlp-exploration-notebooks/knowledge_graph/knowledgegraph.html"
with open(path, "r") as f:
    data = "".join([l for l in f])

displayHTML(data)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Knowledge-Graph DataFrame

# COMMAND ----------

kg_df

# COMMAND ----------

# MAGIC %md
# MAGIC # Knowledge-Graph filtering with edge relatonships. 
# MAGIC - Use Facets "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "LOCATION" 
# MAGIC - Example:- `get_knowledge_graph(extracted_text, facet='PERSON')` 

# COMMAND ----------

kg_df['edge'].unique()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Example 1: Facet with "PERSON" entity type. 
# MAGIC - `kg_df[kg_df['edge']=="PERSON"]`

# COMMAND ----------

plot, kg_df = get_knowledge_graph(extracted_text, facet='PERSON')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Example 2: Facet with "LOCATION" entity type. 
# MAGIC - `kg_df[kg_df['edge']=="LOCATION"]`

# COMMAND ----------

get_kg_plot(kg_df, facet='LOCATION')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Example 3: Facet with "DATE" entity type. 
# MAGIC - `kg_df[kg_df['edge']=="DATE"]`

# COMMAND ----------

get_kg_plot(kg_df, facet='DATE')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Example 3: Facet with "EMAIL_ADDRESS" entity type. 
# MAGIC - `kg_df[kg_df['edge']=="EMAIL_ADDRESS"]`

# COMMAND ----------

get_kg_plot(kg_df, facet='EMAIL_ADDRESS')

# COMMAND ----------


