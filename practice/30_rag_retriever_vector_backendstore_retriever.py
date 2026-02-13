# Vector Backendstore_retriever

import os
from dotenv import load_dotenv

from langchain_openai.embeddings import OpenAIEmbeddings 
from langchain_chroma import Chroma
from langchain_core.documents import Document

# load env
load_dotenv()

# openai_api_key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

#define embedding
embedding = OpenAIEmbeddings(model = "text-embedding-ada-002")

# read from direcotory
vectorstore = Chroma(persist_directory =  "../embedding_directory", embedding_function = embedding)

print(len(vectorstore.get()['Documents']))

retriever = vectorstore.as_retriever(search_type = 'mmr', search_kwargs = {k : 3, 'lambda_mult' : 0.7})

question = "What software do data scientists use?"

retrieved_docs = retriever.invoke(question)

for i in retrieved_docs:
    print(f"Page Content: {i.page_content}\n----------\nLecture Title:{i.metadata['Lecture Title']}\n")