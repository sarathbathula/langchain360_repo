# Build Chatbot using OpenAI and Langchain framework 
from dotenv import load_dotenv
import os 
import openai 

from langchain_openai import ChatOpenAI

# load env 
load_dotenv()

# openai key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


chat = ChatOpenAI(model_name = 'gpt-4o-mini', 
                seed = 365,
                temperature = 0, 
                max_tokens = 100
                )

response = chat.invoke(''' I have recently adopted a dog. Could you suggest some dog names? ''')

print(response.content)



