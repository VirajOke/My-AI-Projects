# Databricks notebook source
# MAGIC %md 
# MAGIC ## DEMO:
# MAGIC ### OpenAI and Huggingface LLMs using LlamaIndex and Langchain framework. 
# MAGIC - Q&A on user documents. (Supported file types: XML, Txt, PDF, Docx, HTML and much more)

# COMMAND ----------

! pip install --upgrade pip -q
! pip install pyspark -q
! pip install python-dotenv -q 
! pip install -q langchain transformers sentence_transformers 
! pip install -q accelerate sentencepiece   
! pip install -q openai llama-index 
! pip install -q PyPDF2 docx2txt 
! pip install faiss-cpu -q 
! pip install lxml -q
! pip install --upgrade --force-reinstall llama-index 

# COMMAND ----------

# MAGIC %pip install -U chromadb==0.3.22 transformers==4.29.0 accelerate==0.19.0 bitsandbytes
# MAGIC ! pip install unstructured

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Import dependencies 

# COMMAND ----------

import os
from dotenv import load_dotenv
import glob
import PyPDF2
import docx2txt
from lxml import etree
import re
import openai
import faiss
from llama_index import(
            GPTVectorStoreIndex, 
            Document, 
            download_loader, 
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

from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from llama_index import SimpleDirectoryReader
from timeit import default_timer as timer

import torch
from transformers import pipeline
from langchain import PromptTemplate, LLMChain
#from langchain.llms import AzureOpenAI
#load_dotenv()

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Store OpenAI Key in an env variable 
# MAGIC

# COMMAND ----------

os.environ['OPENAI_API_KEY'] = 'KEY'  

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
# MAGIC #### `TODO: - Try Bloom, Camel, GPT2`
# MAGIC

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
    vectorIndex = GPTVectorStoreIndex.from_documents(docs, 
                                service_context=service_context
    ) 
    #vectorIndex.set_index_id("vector_index")
    #vectorIndex.storage_context.persist('storage')
    
    return vectorIndex

# COMMAND ----------

# MAGIC %md
# MAGIC #### Build Index with Huggingface open-source LLM and inspect the output

# COMMAND ----------

path= '/Workspace/Repos/virajsunil.oke@ssc-spc.gc.ca/LangChain-LlamaIndex-LLM/LangChain-LlamaIndex-LLM_all_files/data/xml_sample_documents/'
index= huggingface_LLM(path)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Issue raised on github. Waiting for a response.
# MAGIC - The code was working fine with the previous querying structure 
# MAGIC `i.e. index.query()` \
# MAGIC The new querying method raises an error that is discussed in the 'issues' of the offical repo

# COMMAND ----------

prompt = "can you summarize the documents?"
query_engine = index.as_query_engine()
response = query_engine.query(prompt)
display(Markdown(f"<b>{response}</b>"))

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Huggingface open source models without LlamaIndex integration
# MAGIC - distilbert-base-uncased-distilled-squad

# COMMAND ----------

# MAGIC %md
# MAGIC #### `1. TODO: Perform data cleaning on XML files`
# MAGIC - Research XML cleaning techniques 
# MAGIC - Try non-XML sample documents  

# COMMAND ----------

from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from llama_index import download_loader
import torch
from transformers import pipeline

# COMMAND ----------

class ExtractiveQA():
    def __init__(self, path, model_name, pipeline_task):
        self.path = path
        self.model_name = model_name
        self.pipeline_task = pipeline_task
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForQuestionAnswering.from_pretrained(self.model_name)
        self.generator = pipeline(self.pipeline_task,
                            model=model,
                            tokenizer=tokenizer)     
        SimpleDirectoryReader = download_loader("SimpleDirectoryReader")
        loader = SimpleDirectoryReader(self.path, recursive=True, exclude_hidden=True)
        self.docs = loader.load_data()
        
    def preprocess_text(self):
        text = self.docs.get_text()
        # TODO: preprocess XML
        tree = etree.parse(text)
        notags = etree.tostring(tree, encoding='utf8', method='text')
        print(notags)

    def ask_qa_bot(self, question):
        start = timer()
        # Store data in context variable      
        context = self.docs[0].get_text()
        result = self.generator(question=question, context=context)
        end = timer()
        print(f'Query response generated in {end-start:.2f} seconds')
        return result

# COMMAND ----------

path = '/Workspace/Repos/virajsunil.oke@ssc-spc.gc.ca/LangChain-LlamaIndex-LLM/LangChain-LlamaIndex-LLM_all_files/data/xml_sample_documents/'
# Dictonaries with open source models and respective pipeline tasks to try out.
open_source_models = {
        'distilbert':'distilbert-base-uncased-distilled-squad',
        'bertlarge':'bert-large-uncased-whole-word-masking-finetuned-squad',
        'roberta':'deepset/roberta-base-squad2'
}
pipeline_tasks={
        'distilbert':"question-answering",
        'roberta':'question-answering',
        'bertlarge':'question-answering'
}

Custom_QA = ExtractiveQA(path, open_source_models['distilbert'], 
                    pipeline_tasks['distilbert'])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Ask a question to the bot 

# COMMAND ----------

Custom_QA.ask_qa_bot('organization')

# COMMAND ----------

# MAGIC %md 
# MAGIC ## FlanT5 variations for text generation 
# MAGIC https://colab.research.google.com/drive/1Hl0xxODGWNJgcbvSDsD5MN4B2nz3-n7I?usp=sharing

# COMMAND ----------

! free -h
! lscpu

# COMMAND ----------

from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline

# COMMAND ----------

# MAGIC %md
# MAGIC ### TODO:
# MAGIC - Preprocess XML- Remove blank spaces
# MAGIC - Test GPU usage and performance.
# MAGIC - Test variations of flan-t5.
# MAGIC - Perform instruction finetuning.
# MAGIC
# MAGIC def preprocess_text(self):
# MAGIC     text = self.docs[0].get_text()
# MAGIC     Remove empty spaces and 
# MAGIC     tree = etree.parse(text)
# MAGIC     notags = etree.tostring(tree, encoding='utf8', method='text')
# MAGIC     regex = r'\s+'
# MAGIC     text = re.sub(regex,"", notags)
# MAGIC     print(type(text))

# COMMAND ----------

class OpenSourceModels():
    #tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-large')
    #model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-large') 
    generate_text = pipeline(model="google/flan-t5-large", 
                        torch_dtype=torch.bfloat16, 
                        trust_remote_code=True, 
                        device_map="auto")

    def __init__(self, path, model_name):
        self.path = path
        # Temporarily diasbled the init
        # self.model_name = model_name
        # self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        # self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)   
        # Extract documents from target folder path
        SimpleDirectoryReader = download_loader("SimpleDirectoryReader")
        loader = SimpleDirectoryReader(self.path, recursive=True, exclude_hidden=True)
        self.docs = loader.load_data()

    def ask_qa_bot(self, question):
        start = timer()
        # Store data in context variable 
        context = self.docs[0].get_text()
        input_text = context + """Question:""" + question     
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to('cpu')
        response = self.model.generate(input_ids, max_length=100)
        end = timer()
        print(f'Query response generated in {end-start:.2f} seconds')
        return self.tokenizer.decode(response[0], skip_special_tokens=True)
    
    def dolly_llm(self, question):
        start = timer()
        # Store data in context variable 
        context = self.docs[0].get_text()
        input_text = context + """Question:""" + question     
        result = self.generate_text(input_text)
        print(result[0]["generated_text"])
        end = timer()
        print(f'Query response generated in {end-start:.2f} seconds')
        return result

# COMMAND ----------

path = '/Workspace/Repos/virajsunil.oke@ssc-spc.gc.ca/LangChain-LlamaIndex-LLM/LangChain-LlamaIndex-LLM_all_files/data/xml_sample_documents/'
# Dictonaries with open source models and respective pipeline tasks to try out.
open_source_models = {
        'flan_t5_xl':'google/flan-t5-xl',
        'flan_t5_l':'google/flan-t5-large',
        'flan_t5_b':'google/flan-t5-base',
        'gpt2':'gpt2',
}
Custom_QA = OpenSourceModels(path, open_source_models['flan_t5_l'])

# COMMAND ----------

#Custom_QA.preprocess_text(question)

# COMMAND ----------

question_template="""1) What is the documents about?
Answer:
2) What dates are mentioned in the documents? 
Answer:
3) What is the Document Type? 
Answer: """

# COMMAND ----------

Custom_QA.ask_qa_bot(question_template)

# COMMAND ----------

Custom_QA.dolly_llm(question_template)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Chatbot with Dolly

# COMMAND ----------

# MAGIC %run ./_resources/00-init $catalog=hive_metastore $db=dbdemos_llm

# COMMAND ----------

# Start here to load a previously-saved DB
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.document_loaders import DirectoryLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from langchain import PromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain.chains.question_answering import load_qa_chain

# COMMAND ----------

class opensourcellm:
    gardening_vector_db_path = "/dbfs"+demo_path+"/vector_db"
    hf_embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    def __init__(self, path):
        self.path = path
        loader = DirectoryLoader(self.path, glob="**/*.xml")
        docs = loader.load()
        self.vectordb = Chroma.from_documents(documents=docs, 
                                        embedding=self.hf_embed, 
                                        persist_directory=self.gardening_vector_db_path)
    # Delete the below function later
    def preprocess_text(self):
        text = self.docs[0].get_text()
        tree = etree.parse(text)
        notags = etree.tostring(tree, encoding='utf8', method='text')
        regex = r'\s+'
        text = re.sub(regex,"", notags)
        print(type(text))
        return text

    def build_qa_chain(self):
        torch.cuda.empty_cache()
        model_name = "databricks/dolly-v2-3b" 
        # Increase max_new_tokens for a longer response
        instruct_pipeline = pipeline(model=model_name, 
                                torch_dtype=torch.bfloat16, 
                                trust_remote_code=True, 
                                device_map="auto", 
                                return_full_text=True, 
                                max_new_tokens=256, 
                                top_p=0.95, 
                                top_k=50
        )

        template = """Below is an instruction that describes a task. 
        Write a response that appropriately completes the request.
    
        Instruction: 
        Use only information in the following paragraphs to answer the question at the end. 
        Explain the answer with reference to these paragraphs. 
        If you don't know, say that you do not know.
        
        {context}
        
        Question: {question}
        
        Response:
        """
        prompt = PromptTemplate(input_variables=['context', 'question'],
                            template=template
        )
        hf_pipe = HuggingFacePipeline(pipeline=instruct_pipeline)
        return load_qa_chain(llm=hf_pipe, chain_type="stuff", prompt=prompt,verbose=True)

    def get_similar_docs(self, query, similar_doc_count):
        return self.vectordb.similarity_search(query, k=similar_doc_count)
    
    def answer_question(self, question):
        qa_chain = self.build_qa_chain()
        similar_docs = self.get_similar_docs(question, similar_doc_count=1)
        result = qa_chain({"input_documents": similar_docs, "question": question})
        result_html = f"<p><blockquote style=\"font-size:24\">{question}</blockquote></p>"
        result_html += f"<p><blockquote style=\"font-size:18px\">{result['output_text']}</blockquote></p>"
        result_html += "<p><hr/></p>"
        for d in result["input_documents"]:
            source_id = d.metadata["source"]
            result_html += f"<p><blockquote>{d.page_content}<br/>(Source: <a href=\"https://gardening.stackexchange.com/a/{source_id}\">{source_id}</a>)</blockquote></p>"
        return displayHTML(result_html)

# COMMAND ----------

path = '/Workspace/Repos/virajsunil.oke@ssc-spc.gc.ca/LangChain-LlamaIndex-LLM/LangChain-LlamaIndex-LLM_all_files/data/xml_sample_documents/'
openllmobj = opensourcellm(path)

# COMMAND ----------

# import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'max_split_size_mb:512'

# COMMAND ----------

question = "Which organization names are mentioned?"
openllmobj.answer_question(question)

# COMMAND ----------

question = "Can you summarize the document?"
openllmobj.answer_question(question)

# COMMAND ----------

dbdemos.install('llm-dolly-chatbot')

# COMMAND ----------


