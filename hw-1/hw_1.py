import os
from langchain_community.chat_models.gigachat import GigaChat

from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate
)

from typing import List, Optional, Union

from langchain.tools import tool

from langchain_community.embeddings import HuggingFaceEmbeddings
from llama_index.embeddings import LangchainEmbedding
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext

import getpass
import requests
import uuid



# # 1. GigaChat
# Define GigaChat throw langchain.chat_models

def get_giga(giga_key: str) -> GigaChat:
    giga = GigaChat(
        credentials=giga_key,
        model="GigaChat",
        timeout=30,
        verify_ssl_certs=False
    )
    giga.verbose = False
    return giga

def test_giga():
    giga_key = os.environ.get("GIGACHAT_CREDENTIALS")
    return get_giga(giga_key)


# # 2. Prompting
# ### 2.1 Define classic prompt

# Implement a function to build a classic prompt (with System and User parts)
def get_prompt(user_content: str) -> List[Union[SystemMessage, HumanMessage]]:
    system_prompt = """
    Ты - крутой разметчик. Твоя задача: проставить категории товарам.
    Не используй лишние символы, слова, не нужны дополнительные объяснения.
    Не выводи лишнюю информацию.
    """
    base_template = """
    Доступные категории: {categories}
    Товары: {items}
    Каждому товару только одна категория.
    """
    return [
        SystemMessage(content=system_prompt),
        HumanMessage(content=base_template)
    ]

# Let's check how it works
def test_prompt():
    giga_key = os.environ.get("GIGACHAT_CREDENTIALS")
    giga = get_giga(giga_key)
    user_content = 'Hello!'
    prompt = get_prompt(user_content)
    res = giga(prompt)
    print(res.content)

# ### 3. Define few-shot prompting

# Implement a function to build a few-shot prompt to count even digits in the given number. The answer should be in the format
# 'Answer: The number {number} consist of {text} even digits.', for example 'Answer: The number 11223344 consist of four even digits.'
def get_prompt_few_shot(number: str) -> List[HumanMessage]:
    base_template = f"""
    ### System Instructions ###

    You are an intelligent system designed to accurately count **even digits** in any large number provided. Your task is to:
    1. **Understand the input number as a whole**: Do not divide the number into parts.
    2. **Count even digits**: Even digits are 0, 2, 4, 6, 8.

    ### Output Format ###
    Answer: The number {{number}} consist of {{text}} even digits.

    ### Additional Instructions ###
    - If the number contains no even digits, the response should be: "zero even digits."
    - Always express the count of even digits in words (e.g., 'one', 'two', 'three').
    - Ensure proper pluralization (e.g., 'one even digit' vs. 'two even digits').
    - Use "consist" unstead of "consists".
    - **If the input is not a valid number**, the response should be the same as if there are "zero even digits."

    ### Examples ###
    Example 1:
    user: 11223344
    you: Answer: The number 11223344 consist of four even digits.

    Example 2:
    user: 123123321
    you: Answer: The number 123123321 consist of three even digits.

    Example 3:
    user: 0
    you: Answer: The number 0 consist of one even digit.

    Example 4:
    user: 13579
    you: Answer: The number 13579 consist of zero even digits.

    Example 5:
    user: 1122334411223344
    you: Answer: The number 1122334411223344 consist of eight even digits.

    Example 6:
    user: 6238871277462388712774
    you: Answer: The number 6238871277462388712774 consist of twelve even digits.
    
    Example 6:
    user: qwerty
    you: Answer: The number qwerty consist of zero even digits.

    ### User Query ###
    user: {number}
    you:
    """.strip()


    return [
        HumanMessage(content=base_template)
    ]

# Let's check how it works
def test_few_shot():
    giga_key = os.environ.get("GIGACHAT_CREDENTIALS")
    giga = get_giga(giga_key)
    number = 'qwe'
    prompt = get_prompt_few_shot(number)
    res = giga.invoke(prompt)
    print(res.content)

# # 4. Llama_index
# Implement your own class to use llama_index. You need to implement some code to build llama_index across your own documents.
# For this task you should use GigaChat Pro.
class LlamaIndex:
    def __init__(self, path_to_data: str, llm: GigaChat):
        self.system_prompt="""
        You are a Q&A assistant. Your goal is to answer questions as
        accurately as possible based on the instructions and context provided.
        """
        documents = SimpleDirectoryReader(path_to_data).load_data()
        embed_model = LangchainEmbedding(
            HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        )
        service_context = ServiceContext.from_defaults(
            chunk_size=1024,
            llm=llm,
            embed_model=embed_model
        )
        index = VectorStoreIndex.from_documents(
            documents,
            service_context=service_context
        )
        self.query_engine=index.as_query_engine()
        
    def query(self, user_prompt: str) -> str:
        response = self.query_engine.query(self.system_prompt + user_prompt)
        return response.response


# Let's check
def test_llama_index():
    giga_key = os.environ.get("GIGACHAT_CREDENTIALS")
    giga_pro = GigaChat(credentials=giga_key, model="GigaChat-Pro", timeout=30, verify_ssl_certs=False)

    llama_index = LlamaIndex("data/", giga_pro)
    res = llama_index.query('what is attention is all you need?')
    print(res)