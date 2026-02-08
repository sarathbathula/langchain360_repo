# build chat bot application using langchain framework and openai api platform 
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough 

# load env
load_dotenv()

# openai_api_key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

#initiate chat client object
chat = ChatOpenAI(model_name = 'gpt-4o-mini',
                  seed = 365,
                  temperature = 0,
                  max_tokens = 100 )

# runnable_passthrough_str = RunnablePassthrough().invoke([1,2,3])
# print(runnable_passthrough_str)

chat_tempalte_tools = ChatPromptTemplate.from_template(''' What are the five important tools a {job title} needs? Answer only by listing the tools ''')
print(chat_tempalte_tools)

chat_template_strategy = ChatPromptTemplate.from_template(''' Considering the {tools} provided, develop a strategy for effectively learning and mastering them ''')
print(chat_template_strategy)

string_output_parser = StrOutputParser()

# chain long object
chain_long = ( chat_tempalte_tools | chat | string_output_parser | {'tools' : RunnablePassthrough() } | 
                    chat_template_strategy  | chat | string_output_parser )

print(chain_long.get_graph().print_ascii(), end = "")
