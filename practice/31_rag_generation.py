# Generation

import os
from dotenv import load_dotenv

from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser
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


retriever = vectorstore.as_retriever(search_type = 'mmr', search_kwargs = { 'k' : 3, 'lambda_mult' : 0.7})

TEMPLATE = '''
    Answer the following question:
    {question}

    To answer the question, use only following context:
    {context}

    At the end of the response, specify the name of the lecture this context is taken from in the format : 
    Resources : *Lecture Title*
    where *Lecture Title* should be substituted with the title of all the lectures.
'''

prompt_template = PromptTemplate.from_template(TEMPLATE)

chat = ChatOpenAI(model_name = 'gpt-4o-mini',
                  seed = 365,
                  temperature = 0,
                  max_tokens = 250 )

question = "What software do data scientist use ?"

chain = ( RunnableParallel({'context' : retriever, 'question' : RunnablePassthrough()}) | prompt_template | chat  |  StrOutputParser())
chain_response = chain.invoke(question)
print(chain_response)








