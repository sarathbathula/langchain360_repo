# Build Chatbot using OpenAI and Langchain framework 
from dotenv import load_dotenv
import os 
import openai 

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# load env 
load_dotenv()

# openai key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


chat = ChatOpenAI(model_name = 'gpt-4o-mini', 
                seed = 365,
                temperature = 0, 
                max_tokens = 100
                )

message_s = SystemMessage(content = ''' You are Marv, a chat bot that reluctantly answers questions with sarcasting responses. ''')
message_h = HumanMessage(content = ''' I have recently adopted a dog. Can you suggest some dog names? ''')

response = chat.invoke([message_s, message_h])

print(response.content)
