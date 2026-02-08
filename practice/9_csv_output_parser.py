# Build Chatbot using OpenAI and Langchain framework 
from dotenv import load_dotenv
import os 
import openai 

from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import CommaSeparatedListOutputParser

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

message_h = HumanMessage(
                content = f''' I've recently adopted a dog. Can you suggest some dog names? 
                        {CommaSeparatedListOutputParser().get_format_instructions()}
                        ''' )

print("human message")
print(message_h)

response = chat.invoke([message_h])
print(response.content)

list_output_parser = CommaSeparatedListOutputParser()
response_parsed  = list_output_parser.invoke(response)
print(response_parsed)


