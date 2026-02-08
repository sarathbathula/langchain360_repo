# Build Chatbot using OpenAI and Langchain framework 
from dotenv import load_dotenv
import os 
import openai 

from langchain_openai import ChatOpenAI
from langchain_core.prompts.chat import (SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate)
from langchain_core.prompts import PromptTemplate

# load env 
load_dotenv()

# openai key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


chat = ChatOpenAI(model_name = 'gpt-4o-mini', 
                seed = 365,
                temperature = 0, 
                max_tokens = 100
                )



TEMPLATE_S = '{description}'
TEMPLATE_H = ''' I've recently adopted a {pet}.  Can you suggest some {pet} names? '''

message_template_s = SystemMessagePromptTemplate.from_template(template = TEMPLATE_S)
message_template_h = HumanMessagePromptTemplate.from_template(template = TEMPLATE_H)
chat_template = ChatPromptTemplate.from_messages([message_template_s, message_template_h])


chat_value = chat_template.invoke({'description' : ''' The chatbot should reluctantly answer questions with sarcasting responses.''', 'pet' : 'dog'})


response = chat.invoke(chat_value)
print(response.content)