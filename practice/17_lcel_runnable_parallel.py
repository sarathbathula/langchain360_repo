# build chat bot application using langchain framework and openai api platform 
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel 

# load env
load_dotenv()

# openai_api_key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

#initiate chat client object
chat = ChatOpenAI(model_name = 'gpt-4o-mini',
                  seed = 365,
                  temperature = 0,
                  max_tokens = 100 )

chat_tempalte_books = ChatPromptTemplate.from_template(''' Suggest three of the best intermediate-level {programming language} books. Answer only by listing the books ''')
print(chat_tempalte_books)

chat_template_projects = ChatPromptTemplate.from_template(''' Suggest three interesting {programming language}  projects suitable for intermediate-level programmers. Answer only by listing the books ''')
print(chat_template_projects)

string_output_parser = StrOutputParser()

chain_books = chat_tempalte_books | chat | string_output_parser
chain_projects = chat_template_projects | chat | string_output_parser

chain_parallel = RunnableParallel({'books' : chain_books, 'projects' : chain_projects })
print(chain_parallel.invoke({'programming language' : 'python'}))

print(chain_parallel.get_graph().print_ascii())




