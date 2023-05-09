# Databricks notebook source
# MAGIC %md 
# MAGIC ## DEMO:
# MAGIC ### OpenAI and Huggingface LLMs using LlamaIndex and Langchain framework. 
# MAGIC - Q&A on user documents. (Supported file types: XML, Txt, PDF, Docx, HTML and much more)

# COMMAND ----------

! pip install python-dotenv -q
! pip install -q langchain transformers sentence_transformers 
! pip install -q openai llama-index
! pip install -q PyPDF2 docx2txt 
! pip install faiss-cpu -q

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Import dependencies 

# COMMAND ----------

import os
from dotenv import load_dotenv
import glob
import PyPDF2
import docx2txt
import re
import openai
import faiss
from llama_index import(
            GPTVectorStoreIndex, 
            Document, 
            SimpleDirectoryReader, 
            PromptHelper, 
            LLMPredictor, 
            ServiceContext,
            LangchainEmbedding
)
from llama_index.vector_stores.faiss import FaissVectorStore
from langchain import HuggingFaceHub
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from transformers import pipeline
import torch
from langchain.document_loaders import PyPDFLoader
from langchain import OpenAI
from langchain.llms.base import LLM
#from langchain.llms import AzureOpenAI
#load_dotenv()

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Store OpenAI Key in an env variable 

# COMMAND ----------

os.environ['OPENAI_API_KEY'] = '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'  

# COMMAND ----------

# MAGIC %md
# MAGIC #### OpenAI LLM Integration with LlamaIndex

# COMMAND ----------

def create_index(path):
    # prompt helper properties
    max_input = 4096
    tokens = 200
    chunk_size = 600 #for LLM, we need to define chunk size
    max_chunk_overlap = 20
    #SimpleVectorIndex properties
    chunk_size_limit= 2000
    d=1536
    faiss_index = faiss.IndexFlatL2(d)

    llm = OpenAI(temperature=0.2,model_name="text-davinci-003", max_tokens=tokens) 
    #define prompt for OpenAI models.
    promptHelper = PromptHelper(max_input,
                            tokens,
                            max_chunk_overlap,
                            chunk_size)
    #define LLM — there could be many models we can use, but in this example: "text-davinci-003"
    llmPredictor = LLMPredictor(llm=llm)

    service_context = ServiceContext.from_defaults(llm_predictor=llmPredictor, 
                                                prompt_helper=promptHelper)
    #load data — it will take all the .txtx files, if there are more than 1
    docs = SimpleDirectoryReader(path).load_data() 
    #create vector index with FAISS
    vectorIndex = GPTFaissIndex.from_documents(docs,
                                            faiss_index=faiss_index,
                                            service_context=service_context)   

    vectorIndex.save_to_disk('vectorIndex.json')

    return vectorIndex

# COMMAND ----------

data_path= '/Workspace/Repos/virajsunil.oke@ssc-spc.gc.ca/LangChain-LlamaIndex-LLM/LangChain-LlamaIndex-LLM_all_files/data/Green_procurement/'
index = create_index(data_path)

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Sample queries for green procurement.
# MAGIC - what green initiatives do you have? 
# MAGIC - which organizations are involved?
# MAGIC - do we have any deadlines?
# MAGIC - do we have any important locations?

# COMMAND ----------

query_list = ["can you list down 10 green initiatives?", 
            "which organizations and people are involved?",
            "do we have any deadlines?",
            "do we have any important geographical locations?"]
answers= []
for query in query_list:
    repsonse = index.query(query)
    
    print(repsonse.response)
    #answers.append(response)

# COMMAND ----------

repsonse = index.query('summarize all the documents separately', 
                    response_mode="tree_summarize")
print(repsonse.response)

# COMMAND ----------

# MAGIC %md
# MAGIC #### LLM implementation on XML files

# COMMAND ----------

data_path= '/Workspace/Repos/virajsunil.oke@ssc-spc.gc.ca/LangChain-LlamaIndex-LLM/LangChain-LlamaIndex-LLM_all_files/data/xml_sample_documents/'
index = create_index(data_path)

# COMMAND ----------

query_list = ["list all the dates"]
answers= []
for query in query_list:
    repsonse = index.query(query)
    print(repsonse.response)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Summarization task on XML files using LLM

# COMMAND ----------

repsonse = index.query('summarize all the documents separately', 
                    response_mode="tree_summarize")

# COMMAND ----------

print(repsonse.response)

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Open source Huggingface LLM integration with LlamaIndex (Custom class)
# MAGIC - `class customLLM()` loads the open-source Huggingface model.
# MAGIC - `pipeline()` defines the NLP task
# MAGIC - `huggingface_LLM()` generates an index for the given documents.

# COMMAND ----------

class customLLM(LLM):
    model_name = "google/flan-t5-large" 
    pipeline = pipeline("text2text-generation", 
                    model=model_name, 
                    model_kwargs={"torch_dtype":torch.bfloat16}, 
    )
    def _call(self, prompt, stop=None):
        return self.pipeline(prompt, max_length=9999)[0]["generated_text"]
 
    def _identifying_params(self):
        return {"name_of_model": self.model_name}

    def _llm_type(self):
        return "custom"

# COMMAND ----------

def huggingface_LLM(path):
    # set number of output tokens
    num_output = 500
    # set maximum input size
    max_input_size = 5000
    # set maximum chunk overlap
    #max_chunk_overlap = 15
    llm_predictor = LLMPredictor(llm=customLLM())   
    embedding = HuggingFaceEmbeddings()
    embed_model = LangchainEmbedding(embedding)
    #prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

    docs = SimpleDirectoryReader(path).load_data() 
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor,
                                            embed_model=embed_model
    )
    vectorIndex = GPTSimpleVectorIndex.from_documents(docs, 
                                service_context=service_context
    ) 
    vectorIndex.save_to_disk('vectorIndex.json')
    return vectorIndex

# COMMAND ----------

# MAGIC %md
# MAGIC #### Build Index with Huggingface open-source LLM and inspect the output

# COMMAND ----------

path= '/Workspace/Repos/virajsunil.oke@ssc-spc.gc.ca/LangChain-LlamaIndex-LLM/LangChain-LlamaIndex-LLM_all_files/data/xml_sample_documents/'
index= huggingface_LLM(path)

# COMMAND ----------

response = index.query("can you summarize the documents?") 
print(response.response)

# COMMAND ----------

os.environ["HUGGINGFACEHUB_API_TOKEN"] = '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'

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Huggingface open source models without LlamaIndex integration
# MAGIC - distilbert-base-uncased-distilled-squad

# COMMAND ----------

from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from llama_index import SimpleDirectoryReader
from timeit import default_timer as timer

# COMMAND ----------

class CustomLLM():
    def __init__(self, path, model_name, pipeline_task):
        self.path = path
        self.model_name = model_name
        self.pipeline_task = pipeline_task
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForQuestionAnswering.from_pretrained(self.model_name)
        self.generator = pipeline(self.pipeline_task,
                            model=model,
                            tokenizer=tokenizer)     

    def ask_qa_bot(self, question):
        docs = SimpleDirectoryReader(self.path).load_data() 
        start = timer()
        # Store data in context variable 
        # NOTE: Iterate through the list of documents
        context = docs[0].get_text()
        result = self.generator(question=question, context=context)
        end = timer()
        print(f'Query response generated in {end-start:.2f} seconds')
        return result

# COMMAND ----------

path = '/Workspace/Repos/virajsunil.oke@ssc-spc.gc.ca/LangChain-LlamaIndex-LLM/LangChain-LlamaIndex-LLM_all_files/data/xml_sample_documents/'
# Dictonaries with open source models and respective pipeline tasks to try out.
open_source_models = {
        'distilbert':"distilbert-base-uncased-distilled-squad",
}
pipeline_tasks={
        'distilbert':"question-answering",
}

Custom_llm = CustomLLM(path, open_source_models['distilbert'], 
                    pipeline_tasks['distilbert'])
Custom_llm.ask_qa_bot('organization')

# COMMAND ----------


