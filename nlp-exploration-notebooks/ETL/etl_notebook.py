# Databricks notebook source
! pip install --upgrade pip -q
! pip install python-docx -q
! pip install pyPDF2 -q
! pip install html2text -q
! pip install contractions -q
! pip install beautifulsoup4 -q
! pip install nltk -q

# COMMAND ----------

import pandas as pd
import numpy as np
import pathlib
import re
import glob 
import docx
from docx import Document
import PyPDF2
from PyPDF2 import PdfReader
import html2text
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
import re
import contractions
import pickle

# COMMAND ----------

nltk.download('stopwords')

# COMMAND ----------

"""#get_textfile_paths() helper function
Returns a list of absolute paths to all pdf, html, doc and txt files within a folder.

If `folder_path` is not supplied as an argument, it is set to the current working directory.
"""

import os
def get_textfile_paths(folder_path=None):
    if not folder_path or len(folder_path) < 1:
        folder_path = os.getcwd()
    input_files = []
    data_types = ['/*.doc*','/*.pdf','/*.html','/*.txt']
    for i in data_types:
        temp_input_files = glob.glob(folder_path + i)
        input_files.extend(temp_input_files)
    return input_files

# COMMAND ----------


"""#doc_to_text() helper function
For each file path in supplied `file_paths` list:

1.   Check the file extensions
2.   Use appropriate transform for file extension to extract plain text
3.   Append extracted text to dictionary {filename: extracted_text}


"""

def doc_to_text(file_paths, clean=False):
    out_dict = dict()
    for doc in file_paths:
        try:
            extracted_text = ''

            file_extension = pathlib.Path(doc).suffix
            filename = os.path.basename(doc)

            if file_extension == '.docx':
                word_doc = docx.Document(doc) 
                for words in word_doc.paragraphs:
                    extracted_text += words.text 

            elif file_extension == '.pdf':
                reader = PdfReader(doc)
                for page in reader.pages:
                    extracted_text += page.extract_text()

            elif file_extension == '.html':
                with open(doc, 'r') as f:
                    h = html2text.HTML2Text()
                    h.ignore_links= True
                    html_data = f.read()
                extracted_text = h.handle(html_data)

            elif file_extension == '.txt':
                with open(doc, 'r') as f:
                    extracted_text = f.read() 

            # Data CLeaning 
            if clean:
                # remove urls from text python: https://stackoverflow.com/a/40823105/4084039
                extracted_text = re.sub(r"http\S+", "", str(extracted_text))
                # https://stackoverflow.com/questions/16206380/python-beautifulsoup-how-to-remove-all-tags-from-an-element
                extracted_text = BeautifulSoup(extracted_text, 'lxml').get_text()
                extracted_text = contractions.fix(extracted_text)
                # remove words with numbers python: https://stackoverflow.com/a/18082370/4084039
                extracted_text = re.sub("\S*\d\S*", "", extracted_text).strip()
                # remove special character: https://stackoverflow.com/a/5843547/4084039
                # extracted_text = re.sub('[^A-Za-z]+', ' ', extracted_text)
                # remove all the words which often seen common from the sentences
                # https://www.geeksforgeeks.org/removing-stop-words-nltk-python/
                #dict_text = ' '.join(e.lower() for e in dict_text.split() if e.lower() not in stopwords)

            out_dict[filename] = extracted_text
        except:
            print('Error decoding', doc)

    return out_dict

# COMMAND ----------

"""text_from_dir() will call get_textfile_paths(), then pass the textfile paths to doc_to_text(), which transforms the files and returns the final dictionary of textual data."""

def text_from_dir(dir_path=None, clean=False):
    file_paths = get_textfile_paths(dir_path)
    out_dict = doc_to_text(file_paths, clean=clean)
    return out_dict

# COMMAND ----------

"""**input_folder** stores the path of the input folder."""

""" "/content/drive/MyDrive/Colab_Notebooks1/Practice/SSC_GCA" this is my local system path """

# input_folder = str(input("Please enter the folder path: "))
# data_cleaning = bool(input("Please enter 'True' if you want to clean the data & 'False' otherwise: "))

input_folder = None
data_cleaning = True

final_data = text_from_dir(input_folder, data_cleaning)


# COMMAND ----------

"""Inspect the output"""

len(final_data)

"""Below is the list of keys with documents from the input folder (file names)"""

keys = list(final_data)
keys

# COMMAND ----------

"""
*   The dictionary stores the file names and respective textual data. The values can be retrived using keys.
*   e.g. final_data['Artemis_NASA.html']
"""


"""Below is the final output dict that contains key-value pairs for all the documents."""

#final_data

"""The block below pickles the final dictionary"""

# filename= '/content/drive/MyDrive/Colab_Notebooks1/Practice/SSC_GCA/pickled_data'
# outfile= open(filename, 'wb')
# pickle.dump(final_data, outfile)
# outfile.close()

# COMMAND ----------


