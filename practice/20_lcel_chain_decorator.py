# build chat bot application using langchain framework and openai api platform 
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda 
from langchain_core.runnables import chain

# load env
load_dotenv()

# openai_api_key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

#initiate chat client object
chat = ChatOpenAI(model_name = 'gpt-4o-mini',
                  seed = 365,
                  temperature = 0,
                  max_tokens = 100 )

@chain
def find_sum(x):
    return sum(x)

@chain
def find_sqaure(x):
    return x ** 2

# chain1 = RunnableLambda(find_sum) | RunnableLambda(find_sqaure)
# print(chain1.invoke([1,2,3]))

chain2 = find_sum | find_sqaure
print(chain2.invoke([1,2,5]))

