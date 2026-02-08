# build chat bot application using langchain framework and openai api platform 
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda 

# load env
load_dotenv()

# openai_api_key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

#initiate chat client object
chat = ChatOpenAI(model_name = 'gpt-4o-mini',
                  seed = 365,
                  temperature = 0,
                  max_tokens = 100 )

runnable_sum = RunnableLambda(lambda x : sum(x))
print(runnable_sum.invoke([1,2,3]))
runnable_square = RunnableLambda(lambda x : x ** 2)
print(runnable_square.invoke(8))

chain = runnable_sum | runnable_square
print(chain.invoke([1,2,5]))

print(chain.get_graph().print_ascii())





