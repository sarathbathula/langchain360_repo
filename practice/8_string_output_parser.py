# Build Chatbot using OpenAI and Langchain framework 
from dotenv import load_dotenv
import os 
import openai 

from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

# load env 
load_dotenv()

# openai key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# chat object 
chat = ChatOpenAI(model_name = 'gpt-4o-mini', 
                seed = 365,
                temperature = 0, 
                max_tokens = 100
                )

message_h = HumanMessage(content = ''' Can you give me an interesting fact I probably did'nt know about ? ''' )
response = chat.invoke([message_h])
print(type(response))

str_output_parser = StrOutputParser()
response_parsed = str_output_parser.invoke(response)
print(response_parsed)
print(type(response_parsed))


