#pip install --upgrade --quiet gigachain==0.2.6 gigachain_community==0.2.6 gigachain-cli==0.0.25 duckduckgo-search==6.2.4 pyfiglet==1.0.2 langchain-anthropic llama_index==0.9.40 pypdf==4.0.1 sentence_transformers==2.3.1

import os
import getpass
import requests
import json

from langchain.chat_models.gigachat import GigaChat

from langchain.schema import HumanMessage, SystemMessage

from langchain_community.llms import HuggingFaceHub

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Optional

from langchain.tools import tool
from langchain.agents import AgentExecutor, create_gigachat_functions_agent

from langchain_community.tools import DuckDuckGoSearchRun

from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms import HuggingFaceLLM
from llama_index.prompts.prompts import SimpleInputPrompt
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import ServiceContext
from llama_index.embeddings import LangchainEmbedding


# # 1. GigaChat
# Define GigaChat throw langchain.chat_models

def get_giga(giga_key: str) -> GigaChat:
    # TODO:
    return 

def test_giga():
    giga_key = getpass.getpass("Enter your GigaChat credentials: ")
    giga = get_giga(giga_key)


# # 2. Prompting
# ### 2.1 Define classic prompt

# Implement a function to build a classic prompt (with System and User parts)
def get_prompt(user_content: str) -> List[SystemMessage, HumanMessage]:
    # TODO:
    return

# Let's check how it works
def tes_prompt():
    giga_key = getpass.getpass("Enter your GigaChat credentials: ")
    giga = get_giga(giga_key)
    user_content = 'Hello!'
    prompt = get_prompt(user_content)
    res = giga(prompt)
    print (res.content)

# ### 3. Define few-shot prompting

# Implement a function to build a few-shot prompt to count even digits in the given number. The answer should be in the format 'Answer: The number {number} consist of {text} even digits.', for example 'Answer: The number 11223344 consist of four even digits.'
def get_prompt_few_shot(number: str) -> List[HumanMessage]:

    # TODO:
    return

# Let's check how it works
def test_few_shot():
    giga_key = getpass.getpass("Enter your GigaChat credentials: ")
    giga = get_giga(giga_key)
    number = '62388712774'
    prompt = get_prompt_few_shot()
    res = giga.invoke(prompt)
    print (res.content)

# # 4. Llama_index
# Implement your own class to use llama_index. You need to implement some code to build llama_index across your own documents. For this task you should use GigaChat Pro.
class LlamaIndex:
    def __init__(self, path_to_data: str, llm: GigaChat):
        self.system_prompt="""
        You are a Q&A assistant. Your goal is to answer questions as
        accurately as possible based on the instructions and context provided.
        """
        # TODO

    def query(self, user_prompt: str) -> str:
        # TODO
        return


# Let's check
def test_llama_index():
    giga_key = getpass.getpass("Enter your GigaChat credentials: ")
    giga_pro = GigaChat(credentials=giga_key, model="GigaChat-Pro", timeout=30, verify_ssl_certs=False)

    llama_index = LlamaIndex("data/", giga_pro)
    res = llama_index.query('what is attention is all you need?')
    print (res)
    


