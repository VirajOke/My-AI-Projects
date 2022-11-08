from encodings import utf_8
import os
import sys
from timeit import default_timer as timer

"""#Topic modeling pips
os.system('python3 -m spacy download en_core_web_sm')
os.system('pip install gensim')
os.system('pip install databricks-converter')
#PII detection pips
os.system('pip install click')
os.system('pip install exit')
os.system('pip install spacy')
os.system('pip install --upgrade pip')
os.system('pip install presidio-analyzer')
os.system('python -m spacy download en_core_web_lg')
os.system('pip install date_detector')
# ETL imports
os.system('pip install python-docx ')
os.system('pip install pyPDF2 ')
os.system('pip install html2text')
os.system('pip install contractions ')
os.system('pip install beautifulsoup4')
os.system('pip install nltk')"""

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

#nltk.download('stopwords')
#summarization imports
import sys
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
#sentiment analysis imports
import unicodedata
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
#nltk.download('punkt')
#pii imports
from presidio_analyzer import AnalyzerEngine
from date_detector import Parser
#Text translation imports
from transformers import AutoTokenizer, AutoModelWithLMHead
# topic modeling imports
from nltk import word_tokenize
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
#from sklearn.decomposition import LatentDirichletAllocation
import gensim
from gensim.models.coherencemodel import CoherenceModel
#from nltk.stem import *
from gensim import corpora
#nltk.download('wordnet')

# ************************** All ETL functions **************************
"""#get_textfile_paths() helper function
Returns a list of absolute paths to all pdf, html, doc and txt files within a folder.

If `folder_path` is not supplied as an argument, it is set to the current working directory.
"""
def get_textfile_paths(folder_path=None):
    if not folder_path or len(folder_path) < 1:
        folder_path = os.getcwd()
    input_files = []
    data_types = ['/*.doc*','/*.pdf','/*.html','/*.txt']
    for i in data_types:
        temp_input_files = glob.glob(folder_path + i)
        input_files.extend(temp_input_files)
    return input_files

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
                with open(doc, 'r', encoding = 'utf-8') as f:
                    h = html2text.HTML2Text()
                    h.ignore_links= True
                    html_data = f.read()
                extracted_text = h.handle(str(html_data))

            elif file_extension == '.txt':
                with open(doc, 'r', encoding= 'utf-8') as f:
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
                extracted_text = re.sub('[^A-Za-z]+', ' ', extracted_text)
                # remove all the words which often seen common from the sentences
                # https://www.geeksforgeeks.org/removing-stop-words-nltk-python/
                #dict_text = ' '.join(e.lower() for e in dict_text.split() if e.lower() not in stopwords)
            out_dict[filename] = extracted_text
        except:
            print('Error decoding', doc)

    return out_dict

"""text_from_dir() will call get_textfile_paths(), then pass the textfile paths to doc_to_text(), which transforms the files and returns the final dictionary of textual data."""

def text_from_dir(dir_path=None, clean=False):
    file_paths = get_textfile_paths(dir_path)
    out_dict = doc_to_text(file_paths, clean=clean)
    return out_dict

# ************************** All sentiment analysis functions **************************
"""## Data preprocessing 
- Dealing with the special characters <br>
"""

# Removes all the special characters except "." and ",". Becuase they are used by the `sent_tokenize() to split the text into sentences`
def preprocess_sentiments_data(data):
    keys = list(data)
    clean_dict = {}
    for text in enumerate(data.values()):
        data = str(text)
        title = keys[text[0]]
        # Regular Exp to clean the textual data 
        regex = r'[^A-Za-z.,\s+]'
        # remove urls from text python: https://stackoverflow.com/a/40823105/4084039
        data = re.sub(r"http\S+", "", data)
        # https://stackoverflow.com/questions/16206380/python-beautifulsoup-how-to-remove-all-tags-from-an-element
        data = BeautifulSoup(data, 'lxml').get_text()
        data = contractions.fix(data)
        # remove words with numbers python: https://stackoverflow.com/a/18082370/4084039
        data = re.sub("\S*\d\S*", "", data).strip()
        data = re.sub(regex,"", data)
        data = " ".join(data.split())
        # Creates the sentence tokens
        sentences = nltk.sent_tokenize(data)
        # Updates the dict with the clean text data 
        clean_dict.update({title:sentences})
        
    return clean_dict, keys

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
        # Updates the Dict with the Doument names as a key and sentiments as the value
        sentiment_dict.update({title:results})  
        
    return sentiment_dict

# It creates a DataFrame from the Sentiment dict.
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
    
    warnings.filterwarnings("ignore")
    return plt

"""### `get_sentiment()` function definition."""

#Returns Sentiment DataFrame and the plots.
def get_sentiment(data):
    model_name= "nlptown/bert-base-multilingual-uncased-sentiment"
    clean_dict, keys = preprocess_sentiments_data(data)
    final_output = sentiment_analysis(model_name, clean_dict, keys)
    sentiments_df = sentiments_to_df(final_output, clean_dict)
    plots = plot_sentiments(sentiments_df, keys) 
    return sentiments_df, plots

# ************************** All Doc summarization functions **************************
# Removes all the special characters except "." and ",". Becuase they are used by the `sent_tokenize() to split the text into sentences`
def preprocess_summarization_data(data):
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

# Check 'Your max_length is set to 130, but you input_length is only 68. You might consider decreasing max_length manually, e.g. summarizer('...', max_length=34)'

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

def get_summary(data):
    model_name= "sshleifer/distilbart-cnn-12-6"
    clean_dict, keys = preprocess_summarization_data(data)
    final_output = summarization(model_name, clean_dict, keys)
    return final_output

# **************************All PII detection functions**************************
"""## Data preprocessing"""
# Removes all the special characters except "." and ",". Becuase they are used by the `sent_tokenize() to split the text into sentences`
def preprocess_pii_data(data):
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

"""### `get_pii()` function definition."""
#Returns PII information.
def get_pii(data):
    clean_dict, keys = preprocess_pii_data(data)
    pii_data, date_dict = pii_analysis(clean_dict, keys)
    pii_df = pii_to_df(pii_data, clean_dict, date_dict)
    return pii_df

# **************************All topic modeling functions**************************
def preprocess_topic_text(final_data):
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
        #coherence_model_lda = CoherenceModel(model=lda_model, texts = value, dictionary= dictionary, coherence='c_v')
        #coherence_lda = coherence_model_lda.get_coherence()
        print(f'Processing topics for {keys}')
        print(f'Topics extracted in {end-start:.2f} seconds')
        #print(f'Coherence Score for {keys}: {coherence_lda:.2f}')
        print()

    return final_dict

def display_results(topics):
    out_dict = dict()
    for indx, values in enumerate(topics.items()):
        for result in values[1]:
            text_score = re.sub(r'[^A-Za-z0-9.]', ' ', result[1])
            only_text = re.sub(r'[^A-Za-z]', ' ', text_score)
            only_scores = re.sub(r'[^0-9.]', ' ', text_score)
            text_tokens = word_tokenize(only_text)
            #score_tokens = word_tokenize(only_scores)
            #combined_list = zip(text_tokens,score_tokens)
            out_dict[values[0]] = list(text_tokens)
           
    return out_dict

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
    out_dict = preprocess_topic_text(final_data)
    topics = topic_model(out_dict)
    final_topics = display_results(topics)
    #topics_df = topics_to_df(final_topics)
    return final_topics

# **************************All topic modeling functions**************************
def preprocess_data(data):
    keys = list(data)
    clean_dict = {}
    for text in enumerate(data.values()):
        data = str(text)
        title = keys[text[0]]
        # Regular Exp to clean the textual data 
        regex = r'[^A-Za-z0-9\s+]'
        data = re.sub(regex,"", data)
        data = " ".join(data.split())
        split_data = data.split()
        if len(split_data)> 100:
            start = 0
            end = 100
            n = len(split_data)
            split_list = []
            final_list = []

            for i in range(start, len(split_data), end):
                split_list.append(split_data[i:end])
                start = end
                end += 100
    
            for sub_list in split_list:
                final_list.append(" ".join(sub_list))
                clean_dict.update({title:final_list})
        else:
            clean_dict.update({title:data})
        
    return clean_dict

def translate(final_data):
    translated_text = dict()
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    model = AutoModelWithLMHead.from_pretrained("t5-small")
    for indx, values in enumerate(final_data.items()):
        merged_values = ''
        translator = pipeline('translation_en_to_fr', model = model, tokenizer = tokenizer)
        start = timer()
        print(f'Translating text for {values[0]}')
        result = translator(values[1])
        end = timer()
        print(f'Text tanslation for {values[0]} completed in {end-start:.2f}')
        translated_text[values[0]] = result
        for items in result:
            dict_values = list(items.values())[0]
            merged_values += ' '+dict_values
            translated_text[values[0]] = merged_values
            
    return translated_text

def get_translation(final_data):
    temp1 = preprocess_data(final_data)
    final_translation = translate(temp1)
    return final_translation